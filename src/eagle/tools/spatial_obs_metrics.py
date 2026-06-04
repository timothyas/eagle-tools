"""
Compute spatial maps of metrics against observations.

This workflow is the spatial analogue of ``obs_metrics.py``. Rather than
reducing the spatial dimensions to a single number per (forecast hour, initial
condition), it keeps a spatial grid and accumulates a running sum over initial
conditions (analogous to ``spatial.py`` with ``keep_t0=False``).

Observation locations are heterogeneous in time, so for each initial condition
the forecast is interpolated to the observation locations (exactly as in
``obs_metrics.py``), the per-observation error is computed, and those errors are
binned onto the nearest cell of a user-defined target grid. Running sums of the
error, squared error, absolute error, and observation count are kept per cell so
that RMSE, MAE, and bias maps can be formed at the end.
"""
import importlib
import logging

import yaml
import numpy as np
import xarray as xr
import pandas as pd

import xesmf

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

# Running accumulator quantities kept per grid cell.
COMPONENTS = ("sum_err", "sum_sq_err", "sum_abs_err", "count")


def build_target_grid(fds, model_type, d_lat, d_lon=None, lat_bounds=None, lon_bounds=None):
    """Build the target grid the observation errors are binned onto.

    Args:
        fds: A loaded forecast xr.Dataset with ``latitude``/``longitude`` coords.
            Only used (for its spatial extent) when the grid is regional.
        model_type: User model type string. ``"global"`` in the string selects a
            global grid; ``"lam"`` selects a regional grid sized to the forecast.
        d_lat: Latitude grid spacing in degrees.
        d_lon: Longitude grid spacing in degrees. Defaults to ``d_lat``.
        lat_bounds: Optional ``(min, max)`` latitude box (degrees) to restrict the
            grid to. Clipped to the forecast extent.
        lon_bounds: Optional ``(min, max)`` longitude box (degrees, ``[0, 360)``,
            ``min < max``) to restrict the grid to. Clipped to the forecast extent.

    Returns:
        xr.Dataset from xesmf describing the target grid.
    """
    if d_lon is None:
        d_lon = d_lat

    # When the user asks for a region, grid only that box (clipped to the forecast
    # extent) regardless of model type. This is the path for verifying a limited
    # area of an otherwise global forecast.
    if lat_bounds is not None or lon_bounds is not None:
        flat = fds["latitude"].values
        flon = fds["longitude"].values % 360
        lat_lo, lat_hi = lat_bounds if lat_bounds is not None else (
            float(np.nanmin(flat)), float(np.nanmax(flat)))
        lon_lo, lon_hi = lon_bounds if lon_bounds is not None else (
            float(np.nanmin(flon)), float(np.nanmax(flon)))
        lat_lo = max(lat_lo, float(np.nanmin(flat)))
        lat_hi = min(lat_hi, float(np.nanmax(flat)))
        lon_lo = max(lon_lo, float(np.nanmin(flon)))
        lon_hi = min(lon_hi, float(np.nanmax(flon)))
        if not (lat_lo < lat_hi and lon_lo < lon_hi):
            raise ValueError(
                "Requested region does not overlap the forecast extent "
                f"(lat {lat_lo}-{lat_hi}, lon {lon_lo}-{lon_hi})"
            )
        return xesmf.util.grid_2d(lon_lo, lon_hi, d_lon, lat_lo, lat_hi, d_lat)

    if "global" in model_type:
        return xesmf.util.grid_global(d_lon, d_lat, cf=True, lon1=360)
    elif "lam" in model_type:
        lon = fds["longitude"].values % 360
        lat = fds["latitude"].values
        return xesmf.util.grid_2d(
            float(np.nanmin(lon)), float(np.nanmax(lon)), d_lon,
            float(np.nanmin(lat)), float(np.nanmax(lat)), d_lat,
        )
    raise NotImplementedError(f"model_type '{model_type}' must contain 'global' or 'lam'")


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


def _grid_centers(grid):
    """Return ascending 1-D latitude and longitude cell centers as DataArrays.

    Handles both the 2-D ``lat``/``lon`` from ``xesmf.util.grid_2d`` (latitude
    along axis 0, longitude along axis 1) and the 1-D coordinates from
    ``xesmf.util.grid_global(cf=True)``.
    """
    lat = grid["lat"].values
    lon = grid["lon"].values
    lat = lat[:, 0] if lat.ndim == 2 else lat
    lon = lon[0, :] if lon.ndim == 2 else lon
    if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)):
        raise ValueError("Target grid centers must be strictly ascending")
    return (
        xr.DataArray(lat, dims="latitude", coords={"latitude": lat}),
        xr.DataArray(lon, dims="longitude", coords={"longitude": lon}),
    )


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


def _grid_errors(err, snap_lat, snap_lon, lat, lon):
    """Aggregate per-observation errors onto the target grid.

    Each observation has already been snapped to its nearest grid-cell center
    (``snap_lat``/``snap_lon``). Observations sharing a cell are summed, and
    cells with no observations are filled with zero so the result can be added
    into a running accumulator.

    Args:
        err: xr.DataArray of forecast-minus-obs over the ``locations`` dim. Non
            finite entries (e.g. obs outside the forecast domain) are ignored.
        snap_lat, snap_lon: nearest grid-cell centers per observation.
        lat, lon: 1-D DataArrays of grid-cell centers to lay the result out on.

    Returns:
        xr.Dataset with the COMPONENTS as data variables on (latitude, longitude).
    """
    # Drop coords carried over from the interpolation (e.g. the scalar forecast
    # ``time`` and the obs-point ``lat``/``lon``); only the snapped grid centers
    # assigned below should ride along.
    err = err.reset_coords(drop=True)
    valid = err.notnull()
    ds = xr.Dataset(
        {
            "sum_err": err,
            "sum_sq_err": err**2,
            "sum_abs_err": np.abs(err),
            "count": valid.astype(float),
        },
        coords={
            "latitude": ("locations", snap_lat),
            "longitude": ("locations", snap_lon),
        },
    )
    ds = ds.set_index(locations=["latitude", "longitude"]).groupby("locations").sum()
    # unstack fills the "cross" cells (a present latitude x a present longitude
    # with no observation there) -- fill those with 0, not NaN, or they would
    # poison the running accumulator (NaN + x = NaN) and drop later obs there.
    ds = ds.unstack("locations", fill_value=0.0)
    return ds.reindex(latitude=lat, longitude=lon, fill_value=0.0)


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


def _finalize(total, variable_map):
    """Turn accumulated sums into rmse/mae/bias/count datasets, by variable.

    Variables are grouped by base name, stacking levels for upper-air variables,
    mirroring ``obs_metrics``.

    Returns:
        dict mapping metric name -> xr.Dataset, each variable having dims
        (fhr, [level,] latitude, longitude).
    """
    metrics = ("rmse", "mae", "bias", "count")

    base_groups = {}
    for forecast_var, vinfo in variable_map.items():
        base_groups.setdefault(vinfo["base_name"], {})[vinfo["level"]] = forecast_var

    containers = {}
    for metric in metrics:
        data_vars = {}
        for base_name, level_dict in base_groups.items():
            if None in level_dict:
                data_vars[base_name] = _metric(total[level_dict[None]], metric)
            else:
                levels = sorted(level_dict.keys())
                arrays = [
                    _metric(total[level_dict[lvl]], metric).expand_dims({"level": [lvl]})
                    for lvl in levels
                ]
                data_vars[base_name] = xr.concat(arrays, dim="level")
        ds = xr.Dataset(data_vars)
        ds = ds.assign_coords(
            lead_time=("fhr", pd.to_timedelta(ds["fhr"].values, unit="h"))
        )
        containers[metric] = ds
    return containers


def _reduce_accumulators(total, topo, forecast_var_names, fhr_coord, lat, lon):
    """Sum the per-rank accumulators onto the root rank with MPI.

    Uses a buffer-based ``comm.Reduce`` on each accumulator array (a few hundred
    MB) rather than pickling and gathering whole accumulator dicts, which for
    fine grids would blow past mpi4py's ~2 GiB pickle-message limit and pile
    every rank's data up on root.

    The grid/forecast-hour metadata is broadcast from root so that any rank that
    processed no initial conditions can still take part in the collective with
    matching (zero-filled) buffers.

    Returns:
        On root, the reduced accumulator dict; on other ranks, the (unused)
        local accumulators.
    """
    from mpi4py import MPI

    comm = topo.comm
    forecast_var_names = comm.bcast(forecast_var_names, root=topo.root)
    fhr_coord = comm.bcast(fhr_coord, root=topo.root)
    lat = comm.bcast(lat, root=topo.root)
    lon = comm.bcast(lon, root=topo.root)

    if total is None:  # this rank processed no initial conditions
        template = xr.Dataset(
            {c: xr.zeros_like(lat * lon) for c in COMPONENTS}
        ).expand_dims(fhr=fhr_coord)
        total = {var: template.copy(deep=True) for var in forecast_var_names}

    for var in forecast_var_names:
        for comp in COMPONENTS:
            send = np.ascontiguousarray(total[var][comp].values)
            recv = np.empty_like(send) if topo.is_root else None
            comm.Reduce(send, recv, op=MPI.SUM, root=topo.root)
            if topo.is_root:
                total[var][comp].data[...] = recv
    return total


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
    model_type = config.get("model_type", "global")
    temporal_window = pd.Timedelta(config.get("temporal_window", "30min"))
    max_qc_value = config.get("max_qc_value", 2)
    user_levels = config.get("levels", None)
    d_lat = config["d_lat"]
    d_lon = config.get("d_lon", None)

    # Optional region of interest: restrict the forecast, observations, and target
    # grid to this latitude/longitude box. Longitude is in [0, 360) with min < max
    # (a contiguous box that does not wrap across the 0/360 seam).
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
    logger.info(f"Initial Conditions:\n{dates}")

    # Lazily initialized once the first forecast is loaded
    levels = user_levels
    variable_map = None
    forecast_var_names = None
    lat = lon = None      # 1-D grid-center DataArrays
    fhr_coord = None      # canonical forecast hours
    total = None          # {forecast_var: accumulator xr.Dataset}

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

        # One-time setup once we know the forecast grid, levels, and variables
        if variable_map is None:
            if levels is None and "level" in fds.dims:
                levels = sorted(int(lvl) for lvl in fds["level"].values)
                logger.info(f"Discovered levels from forecast: {levels}")
            variable_map = build_variable_map(config, levels, obs_config=obs_config)
            forecast_var_names = list(variable_map.keys())
            logger.info(f"Variables to verify: {forecast_var_names}")

            lat, lon = _grid_centers(build_target_grid(
                fds, model_type, d_lat, d_lon, lat_bounds=lat_bounds, lon_bounds=lon_bounds))
            fhr_coord = (
                (forecast_valid_times - np.datetime64(t0)) / np.timedelta64(1, "h")
            ).astype(int)
            logger.info(f"Target grid: {lat.size} x {lon.size} (lat x lon)")

            template = xr.Dataset(
                {c: xr.zeros_like(lat * lon) for c in COMPONENTS}
            ).expand_dims(fhr=fhr_coord)
            total = {var: template.copy(deep=True) for var in forecast_var_names}

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

        # Accumulate gridded sums for this initial condition, adding each forecast
        # hour straight into its slice of the running accumulator. (Building up a
        # list of per-hour grids and concatenating instead would roughly double
        # the in-memory footprint for fine grids.)
        for fhr_idx, vtime in enumerate(forecast_valid_times):
            vtimestamp = pd.Timestamp(vtime, tz="UTC")
            if vtimestamp not in aligned:
                continue
            matched_obs = aligned[vtimestamp]

            # Snap each observation to the nearest target-grid cell center
            obs_lat = xr.DataArray(matched_obs["LAT"].values, dims="locations")
            obs_lon = xr.DataArray(matched_obs["LON"].values, dims="locations")
            snap_lat = lat.sel(latitude=obs_lat, method="nearest").values
            snap_lon = lon.sel(longitude=obs_lon, method="nearest").values

            interpolated = _interp_to_obs_locations(fds.sel(time=vtime), matched_obs)

            for varname, vinfo in variable_map.items():
                fda = _extract_forecast_values(interpolated, vinfo)
                obs = xr.DataArray(matched_obs[vinfo["obs_col"]].values, dims="locations")
                gridded = _grid_errors(fda - obs, snap_lat, snap_lon, lat, lon)
                for comp in COMPONENTS:
                    total[varname][comp].data[fhr_idx] += gridded[comp].values

        del fds, obs_df, aligned
        logger.info(f"Done with {st0}")

    logger.info("Done Computing Spatial Observation Verification Metrics")

    # Sum the running accumulators across MPI ranks onto root (serial runs already
    # hold the complete accumulator on the single process)
    if use_mpi:
        logger.info("Reducing accumulators across ranks")
        total = _reduce_accumulators(total, topo, forecast_var_names, fhr_coord, lat, lon)

    if topo.is_root:
        containers = _finalize(total, variable_map)
        output_path = config["output_path"]
        for metric, xds in containers.items():
            fname = f"{output_path}/spatial.{metric}.convobs.{model_type}.nc"
            xds.to_netcdf(fname)
            logger.info(f"Stored result: {fname}")
        logger.info("Done Storing Spatial Observation Verification Metrics")
