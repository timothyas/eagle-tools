"""
Compute spatial maps of metrics against observations.

This workflow is the spatial analogue of ``obs_metrics.py``. Rather than
reducing the spatial dimensions to a single number per (forecast hour, initial
condition), it keeps a per-station accumulation and runs a running sum over
initial conditions (analogous to ``spatial.py`` with ``keep_t0=False``).

Observation locations are heterogeneous in time, so for each initial condition
the forecast is interpolated to the observation locations (exactly as in
``obs_metrics.py``), the per-observation error is computed, and those errors are
aggregated by station. A "station" is just the observation's own location
rounded to ``round_decimals`` decimal degrees (~100 m at the default of 3), so
we stay in observation space rather than binning onto a model grid. Running
sums of the error, squared error, absolute error, and observation count are kept
per station so that RMSE, MAE, and bias can be formed at the end.

Keying on the (rounded) observation location -- rather than snapping to a
model-specific target grid -- means two different forecasts verified against the
same observation network land on an *identical* station axis, so their per-cell
errors can be differenced directly downstream with no regridding. The output has
dims (fhr, [level,] station) with ``latitude``/``longitude`` coordinates riding
along the ``station`` dimension.
"""
import importlib
import logging

import yaml
import numpy as np
import xarray as xr
import pandas as pd

from eagle.tools.obs_metrics import (
    _load_obs_config,
    _get_unit_conversion_funcs,
    _load_forecast,
    _interp_to_obs_locations,
    build_variable_map,
    load_all_observations,
    apply_qc_filter,
    convert_obs_units,
    convert_dewpoint_to_specific_humidity,
    derive_wind_components,
    align_obs_to_forecast_times,
)

logger = logging.getLogger("eagle.tools")

# Running accumulator quantities kept per station.
COMPONENTS = ("sum_err", "sum_sq_err", "sum_abs_err", "count")


def _subset_forecast_to_bounds(fds, lat_bounds, lon_bounds):
    """Restrict a forecast to a latitude/longitude box, padded for interpolation.

    Only applies to forecasts laid out on 1-D ``latitude``/``longitude`` dimension
    coordinates (i.e. a regular global grid). Curvilinear forecasts (e.g. a LAM on
    ``y``/``x``) are returned unchanged -- the observation filter still limits the
    region for those.

    The box is padded by a couple of native grid cells so that the bilinear
    interpolation onto observation points near the edge of the box still has a
    surrounding stencil of forecast cells.
    """
    if "latitude" not in fds.dims or "longitude" not in fds.dims:
        return fds

    def _pad(coord):
        diffs = np.abs(np.diff(np.unique(coord.values)))
        return 2.0 * float(np.median(diffs)) if diffs.size else 0.0

    if lat_bounds is not None:
        lo, hi = lat_bounds
        pad = _pad(fds["latitude"])
        mask = (fds["latitude"] >= lo - pad) & (fds["latitude"] <= hi + pad)
        fds = fds.sel(latitude=fds["latitude"][mask.values])

    if lon_bounds is not None:
        lo, hi = lon_bounds
        pad = _pad(fds["longitude"])
        lon360 = fds["longitude"] % 360
        mask = (lon360 >= lo - pad) & (lon360 <= hi + pad)
        fds = fds.sel(longitude=fds["longitude"][mask.values])

    return fds


def _filter_obs_to_bounds(obs_df, lat_bounds, lon_bounds):
    """Drop observations outside the requested latitude/longitude box.

    Assumes ``LON`` has already been wrapped into ``[0, 360)`` to match the
    forecast convention, and ``lon_bounds`` is a contiguous ``min < max`` box.
    """
    if lat_bounds is not None:
        lo, hi = lat_bounds
        obs_df = obs_df[(obs_df["LAT"] >= lo) & (obs_df["LAT"] <= hi)]
    if lon_bounds is not None:
        lo, hi = lon_bounds
        obs_df = obs_df[(obs_df["LON"] >= lo) & (obs_df["LON"] <= hi)]
    return obs_df


def _station_ids(lat, lon, round_decimals):
    """Per-observation station ids (rounded "lat_lon") and their rounded coords.

    Two observations from the same fixed station report nearly identical
    coordinates, so rounding collapses them onto one station id. Returns the id
    array plus the rounded latitude/longitude arrays (used to attach coordinates
    to the station axis).
    """
    rlat = np.round(lat, round_decimals)
    rlon = np.round(lon, round_decimals)
    sid = np.array([f"{a:.{round_decimals}f}_{o:.{round_decimals}f}"
                    for a, o in zip(rlat, rlon)])
    return sid, rlat, rlon


def _extract_forecast_values(interpolated, vinfo):
    """Per-observation forecast values for a variable, as an xr.DataArray.

    Mirrors the variable handling in ``obs_metrics`` (derived wind speed and
    level selection). Returns a DataArray over the ``locations`` dimension.
    """
    if "forecast_components" in vinfo:
        components = vinfo["forecast_components"]
        if "u" not in components or "v" not in components:
            raise ValueError(
                f"Unknown forecast_components {components}. "
                "Only wind speed (u/v) derivation is currently supported."
            )
        u = interpolated[components["u"]]
        v = interpolated[components["v"]]
        if "level" in u.dims:
            u = u.sel(level=vinfo["level"])
            v = v.sel(level=vinfo["level"])
        return np.sqrt(u**2 + v**2)

    fvals = interpolated[vinfo["base_name"]]
    if "level" in fvals.dims:
        fvals = fvals.sel(level=vinfo["level"])
    return fvals


def _station_sums(err, sid, fhr):
    """Per-station summed accumulator components for one (var, fhr).

    ``err`` is the per-observation forecast-minus-obs array (non-finite entries,
    e.g. obs outside the forecast domain or masked by QC, contribute 0 and are
    not counted). Returns a DataFrame indexed by (station, fhr).
    """
    finite = np.isfinite(err)
    df = pd.DataFrame({
        "station": sid,
        "sum_err": np.where(finite, err, 0.0),
        "sum_sq_err": np.where(finite, err * err, 0.0),
        "sum_abs_err": np.where(finite, np.abs(err), 0.0),
        "count": finite.astype(float),
    })
    g = df.groupby("station", sort=False)[list(COMPONENTS)].sum()
    g["fhr"] = fhr
    return g.set_index("fhr", append=True)


def _metric(ds, name):
    """Form a single metric DataArray from the accumulated sums."""
    count = ds["count"]
    if name == "count":
        return count
    denom = count.where(count > 0)  # NaN where no obs -> metric is NaN there
    if name == "rmse":
        return np.sqrt(ds["sum_sq_err"] / denom)
    if name == "mae":
        return ds["sum_abs_err"] / denom
    if name == "bias":
        return ds["sum_err"] / denom
    raise ValueError(f"unknown metric '{name}'")


def _to_station_dataset(df, station_loc):
    """Convert a (station, fhr)-indexed sums DataFrame to an xr.Dataset.

    The result has dims (station, fhr) with the accumulator COMPONENTS as data
    variables and ``latitude``/``longitude`` coordinates on the station axis.
    (station, fhr) combinations never observed are filled with 0.
    """
    ds = df.sort_index().to_xarray()
    for c in COMPONENTS:
        ds[c] = ds[c].fillna(0.0)
    sids = ds["station"].values
    lat = np.array([station_loc[s][0] for s in sids])
    lon = np.array([station_loc[s][1] for s in sids])
    return ds.assign_coords(latitude=("station", lat), longitude=("station", lon))


def _finalize(total, station_loc, variable_map):
    """Turn accumulated sums into rmse/mae/bias/count datasets, by variable.

    Variables are grouped by base name, stacking levels for upper-air variables,
    mirroring ``obs_metrics``.

    Returns:
        dict mapping metric name -> xr.Dataset, each variable having dims
        (fhr, [level,] station).
    """
    metrics = ("rmse", "mae", "bias", "count")

    station_ds = {var: _to_station_dataset(df, station_loc)
                  for var, df in total.items() if df is not None}

    base_groups = {}
    for forecast_var, vinfo in variable_map.items():
        if forecast_var in station_ds:
            base_groups.setdefault(vinfo["base_name"], {})[vinfo["level"]] = forecast_var

    containers = {}
    for metric in metrics:
        data_vars = {}
        for base_name, level_dict in base_groups.items():
            if None in level_dict:
                data_vars[base_name] = _metric(station_ds[level_dict[None]], metric)
            else:
                levels = sorted(level_dict.keys())
                arrays = [
                    _metric(station_ds[level_dict[lvl]], metric).expand_dims({"level": [lvl]})
                    for lvl in levels
                ]
                data_vars[base_name] = xr.concat(arrays, dim="level")
        ds = xr.Dataset(data_vars)
        ds = ds.assign_coords(
            lead_time=("fhr", pd.to_timedelta(ds["fhr"].values, unit="h"))
        )
        containers[metric] = ds
    return containers


def _merge_station_accumulators(total, station_loc, topo, forecast_var_names):
    """Sum the per-rank station accumulators onto root with MPI.

    The ranks agree on a common (station, fhr) layout, then each variable's
    components are summed with a buffer-based ``comm.Reduce`` on dense
    ``(n_station, n_fhr)`` float arrays (a few MB each). We deliberately avoid
    pickling and gathering the whole per-rank DataFrames: over a long record the
    station set grows large enough that a single ``comm.gather`` of them exceeds
    mpi4py's ~2 GiB pickle-message limit (MPI_ERR_ARG). Only the small
    station-id -> lat/lon maps are gathered.

    Returns:
        On root, the merged (total, station_loc); on other ranks, the inputs.
    """
    from mpi4py import MPI

    comm = topo.comm
    # ranks that processed no initial conditions never learned these
    forecast_var_names = comm.bcast(forecast_var_names, root=topo.root)
    if total is None:
        total = {}

    # global station ordering: gather the small id->lat/lon maps onto root, then
    # broadcast the agreed (sorted) station list so every rank lays out the same.
    gathered_loc = comm.gather(station_loc, root=topo.root)
    if topo.is_root:
        merged_loc = {}
        for loc in gathered_loc:
            merged_loc.update(loc)
        stations = sorted(merged_loc)
    else:
        merged_loc = stations = None
    stations = comm.bcast(stations, root=topo.root)
    sidx = {s: i for i, s in enumerate(stations)}

    # global (canonical) forecast hours
    local_fhrs = set()
    for df in total.values():
        if df is not None:
            local_fhrs |= {int(f) for f in df.index.get_level_values("fhr")}
    fhrs = sorted(set().union(*comm.allgather(local_fhrs)))
    fidx = {f: i for i, f in enumerate(fhrs)}

    ns, nf = len(stations), len(fhrs)
    merged = {}
    for var in forecast_var_names:
        df = total.get(var)
        if df is not None:
            si = df.index.get_level_values("station").map(sidx).to_numpy()
            fi = df.index.get_level_values("fhr").map(lambda f: fidx[int(f)]).to_numpy()
        comps = {}
        for comp in COMPONENTS:
            send = np.zeros((ns, nf), dtype="float64")
            if df is not None:
                send[si, fi] = df[comp].to_numpy()
            recv = np.empty_like(send) if topo.is_root else None
            comm.Reduce(send, recv, op=MPI.SUM, root=topo.root)
            if topo.is_root:
                comps[comp] = recv.reshape(-1)
        if topo.is_root:
            idx = pd.MultiIndex.from_product([stations, fhrs], names=["station", "fhr"])
            merged[var] = pd.DataFrame(comps, index=idx)

    if topo.is_root:
        return merged, merged_loc
    return total, station_loc


def main(config):
    """Compute spatial maps of metrics against observations.

    See ``eagle-tools spatial-obs-metrics --help`` or cli.py for help.
    """
    if isinstance(config, str):
        from eagle.tools.utils import setup
        config = setup(config, "spatial_obs_metrics")

    topo = config["topo"]
    use_mpi = config["use_mpi"]
    obs_config = _load_obs_config()
    dataset_registry = obs_config["dataset_registry"]
    unit_conversion_funcs = _get_unit_conversion_funcs(obs_config["gravity"])

    # Config options
    model_type = config.get("model_type", "global")  # filename tag only
    temporal_window = pd.Timedelta(config.get("temporal_window", "30min"))
    max_qc_value = config.get("max_qc_value", 2)
    # Optional subset of forecast hours to verify (default: all in the forecast).
    # Restricting this also shrinks the per-IC observation-loading window.
    keep_fhrs = config.get("fhrs", None)
    if keep_fhrs is not None:
        keep_fhrs = np.asarray([int(f) for f in keep_fhrs])
    user_levels = config.get("levels", None)
    # Decimal degrees to round obs locations to when forming station ids; 3 ~ 100 m.
    round_decimals = config.get("round_decimals", 3)

    # Optional region of interest: restrict the forecast and observations to this
    # latitude/longitude box. Longitude is in [0, 360) with min < max (a contiguous
    # box that does not wrap across the 0/360 seam).
    lat_bounds = config.get("latitude_bounds", None)
    lon_bounds = config.get("longitude_bounds", None)
    if lat_bounds is not None:
        lat_bounds = (float(lat_bounds[0]), float(lat_bounds[1]))
        if not lat_bounds[0] < lat_bounds[1]:
            raise ValueError(f"latitude_bounds must be (min, max) with min < max, got {lat_bounds}")
    if lon_bounds is not None:
        lon_bounds = (float(lon_bounds[0]) % 360, float(lon_bounds[1]) % 360)
        if not lon_bounds[0] < lon_bounds[1]:
            raise ValueError(
                "longitude_bounds must be a contiguous (min, max) box in [0, 360) "
                f"with min < max (wrapping across 0/360 is not supported), got {lon_bounds}"
            )
    if lat_bounds is not None or lon_bounds is not None:
        logger.info(f"Restricting to region: latitude={lat_bounds}, longitude={lon_bounds}")

    if config.get("n_members", 1) > 1:
        raise NotImplementedError("spatial-obs-metrics supports deterministic forecasts only")

    # Derived variables (e.g. wind_speed) don't exist in raw forecast datasets
    derived_variables = obs_config.get("derived_variables", {})
    rename_path = importlib.resources.files("eagle.tools.config") / "rename.yaml"
    with rename_path.open("r") as f:
        rdict = yaml.safe_load(f)
    reverse_rename = {v: k for k, v in rdict.items()}
    derived_var_names = set(derived_variables.keys())
    for ln in list(derived_var_names):
        if ln in reverse_rename:
            derived_var_names.add(reverse_rename[ln])

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])
    n_dates = len(dates)
    n_batches = int(np.ceil(n_dates / topo.size))

    logger.info("Spatial Observation Verification")
    logger.info(f"Datasets: {list(dataset_registry.keys())}")
    logger.info(f"Temporal window: +/- {temporal_window}")
    logger.info(f"Max QC value: {max_qc_value}")
    logger.info(f"Station rounding: {round_decimals} decimal degrees")
    logger.info(f"Initial Conditions:\n{dates}")

    # Lazily initialized once the first forecast is loaded
    levels = user_levels
    variable_map = None
    forecast_var_names = None
    total = None              # {forecast_var: (station, fhr) sums DataFrame}
    station_loc = {}          # station id -> (latitude, longitude)

    for batch_idx in range(n_batches):

        date_idx = (batch_idx * topo.size) + topo.rank
        if date_idx + 1 > n_dates:
            break

        t0 = dates[date_idx]
        st0 = t0.strftime("%Y-%m-%dT%H")
        logger.info(f"Processing {st0}")

        fds = _load_forecast(config, t0, 0, levels, derived_var_names=derived_var_names)
        if lat_bounds is not None or lon_bounds is not None:
            fds = _subset_forecast_to_bounds(fds, lat_bounds, lon_bounds)
        forecast_valid_times = fds["time"].values
        fhrs = (
            (forecast_valid_times - np.datetime64(t0)) / np.timedelta64(1, "h")
        ).astype(int)
        if keep_fhrs is not None:
            keep = np.isin(fhrs, keep_fhrs)
            forecast_valid_times = forecast_valid_times[keep]
            fhrs = fhrs[keep]

        # One-time setup once we know the forecast grid, levels, and variables
        if variable_map is None:
            if levels is None and "level" in fds.dims:
                levels = sorted(int(lvl) for lvl in fds["level"].values)
                logger.info(f"Discovered levels from forecast: {levels}")
            variable_map = build_variable_map(config, levels, obs_config=obs_config)
            forecast_var_names = list(variable_map.keys())
            logger.info(f"Variables to verify: {forecast_var_names}")
            total = {var: None for var in forecast_var_names}

        # Load observations over the full valid-time range (padded by the window)
        time_start = pd.Timestamp(forecast_valid_times[0], tz="UTC") - temporal_window - pd.Timedelta("24h")
        time_end = pd.Timestamp(forecast_valid_times[-1], tz="UTC") + temporal_window + pd.Timedelta("24h")
        obs_df = load_all_observations((time_start, time_end), variable_map, dataset_registry)

        obs_df = apply_qc_filter(obs_df, variable_map, max_qc_value=max_qc_value)
        obs_df = convert_obs_units(obs_df, variable_map, unit_conversion_funcs)
        obs_df = convert_dewpoint_to_specific_humidity(obs_df, variable_map)
        obs_df = derive_wind_components(obs_df, variable_map)
        obs_df["LON"] = obs_df["LON"] % 360
        if lat_bounds is not None or lon_bounds is not None:
            obs_df = _filter_obs_to_bounds(obs_df, lat_bounds, lon_bounds)

        aligned = align_obs_to_forecast_times(obs_df, forecast_valid_times, temporal_window)

        # Aggregate this initial condition's per-station sums, then fold them into
        # the running total (a union over stations, since the observed set varies
        # between initial conditions).
        ic_frames = {var: [] for var in forecast_var_names}
        for fhr_idx, vtime in enumerate(forecast_valid_times):
            vtimestamp = pd.Timestamp(vtime, tz="UTC")
            if vtimestamp not in aligned:
                continue
            matched_obs = aligned[vtimestamp]

            sid, rlat, rlon = _station_ids(
                matched_obs["LAT"].values, matched_obs["LON"].values, round_decimals)
            for s, a, o in zip(sid, rlat, rlon):
                station_loc.setdefault(s, (float(a), float(o)))

            interpolated = _interp_to_obs_locations(fds.sel(time=vtime), matched_obs)

            for varname, vinfo in variable_map.items():
                fda = _extract_forecast_values(interpolated, vinfo)
                obs = xr.DataArray(matched_obs[vinfo["obs_col"]].values, dims="locations")
                err = (fda - obs).values
                ic_frames[varname].append(_station_sums(err, sid, int(fhrs[fhr_idx])))

        for varname in forecast_var_names:
            if not ic_frames[varname]:
                continue
            ic = pd.concat(ic_frames[varname]).groupby(level=["station", "fhr"]).sum()
            total[varname] = ic if total[varname] is None else total[varname].add(ic, fill_value=0.0)

        del fds, obs_df, aligned
        logger.info(f"Done with {st0}")

    logger.info("Done Computing Spatial Observation Verification Metrics")

    # Sum the running accumulators across MPI ranks onto root (serial runs already
    # hold the complete accumulator on the single process)
    if use_mpi:
        logger.info("Reducing accumulators across ranks")
        total, station_loc = _merge_station_accumulators(
            total, station_loc, topo, forecast_var_names)

    if topo.is_root:
        containers = _finalize(total, station_loc, variable_map)
        output_path = config["output_path"]
        for metric, xds in containers.items():
            fname = f"{output_path}/spatial.{metric}.convobs.{model_type}.nc"
            xds.to_netcdf(fname)
            logger.info(f"Stored result: {fname}")
        logger.info("Done Storing Spatial Observation Verification Metrics")
