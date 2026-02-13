import logging

import numpy as np
import xarray as xr
import pandas as pd

import nnja_ai

from eagle.tools.data import open_anemoi_inference_dataset, open_forecast_zarr_dataset
from eagle.tools.metrics import postprocess

logger = logging.getLogger("eagle.tools")

GRAVITY = 9.80665

UNIT_CONVERSIONS = {
    "gp_to_gph": lambda x: x / GRAVITY,
}

DEFAULT_VARIABLES = {
    "t": {
        "obs_var": "TMDB",
        "obs_qc_var": "QMAT",
        "levels": [850],
    },
    "gh": {
        "obs_var": "GP10",
        "obs_qc_var": "QMGP",
        "levels": [500],
        "unit_conversion": "gp_to_gph",
    },
}


def build_variable_map(config):
    """Expand config variables dict into a flat map keyed by forecast variable name.

    Returns:
        dict: e.g. {
            "t_850": {
                "obs_col": "TMDB_PRLC85000",
                "obs_qc_col": "QMAT_PRLC85000",
                "unit_conversion": None,
            },
            "gh_500": {
                "obs_col": "GP10_PRLC50000",
                "obs_qc_col": "QMGP_PRLC50000",
                "unit_conversion": "gp_to_gph",
            },
        }
    """
    variables = config.get("variables", DEFAULT_VARIABLES)
    variable_map = {}
    for base_name, vinfo in variables.items():
        obs_var = vinfo["obs_var"]
        obs_qc_var = vinfo["obs_qc_var"]
        conversion = vinfo.get("unit_conversion", None)
        for level in vinfo["levels"]:
            forecast_var = f"{base_name}_{level}"
            prlc_suffix = f"PRLC{level * 100}"
            variable_map[forecast_var] = {
                "base_name": base_name,
                "level": level,
                "obs_col": f"{obs_var}_{prlc_suffix}",
                "obs_qc_col": f"{obs_qc_var}_{prlc_suffix}",
                "unit_conversion": conversion,
            }
    return variable_map


def load_observations(config, time_range, variable_map):
    """Load observation data from nnja_ai DataCatalog.

    Args:
        config: Config dict with obs_dataset key.
        time_range: (start, end) tuple of pd.Timestamps for time selection.
        variable_map: Output of build_variable_map.

    Returns:
        pd.DataFrame with LAT, LON, OBS_TIMESTAMP, and obs/QC columns.
    """
    obs_dataset = config.get("obs_dataset", "conv-adpupa-NC002001")

    columns = ["LAT", "LON", "OBS_TIMESTAMP"]
    for vinfo in variable_map.values():
        columns.append(vinfo["obs_col"])
        columns.append(vinfo["obs_qc_col"])
    columns = list(dict.fromkeys(columns))  # deduplicate, preserve order

    dc = nnja_ai.DataCatalog()
    ds = dc[obs_dataset]
    try:
        subds = ds.sel(
            time=slice(str(time_range[0]), str(time_range[1])),
            variables=columns,
        )
        obs_df = subds.load_dataset()
    except nnja_ai.exceptions.EmptyTimeSubsetError:
        logger.warning(f"No observations found for time range {time_range[0]} to {time_range[1]}")
        obs_df = pd.DataFrame(columns=columns)
    logger.info(f"Loaded {len(obs_df)} observations from {obs_dataset}")
    return obs_df


def apply_qc_filter(obs_df, variable_map, max_qc_value=2):
    """Apply per-variable QC filtering.

    Masks obs values to NaN where QC is non-NaN AND > max_qc_value.
    NaN QC means not flagged -> keep. 0-2 = good -> keep. 3+ = suspect/rejected -> mask.
    """
    for forecast_var, vinfo in variable_map.items():
        obs_col = vinfo["obs_col"]
        qc_col = vinfo["obs_qc_col"]
        if qc_col in obs_df.columns and obs_col in obs_df.columns:
            qc_vals = obs_df[qc_col]
            bad = qc_vals.notna() & (qc_vals > max_qc_value)
            n_rejected = bad.sum()
            if n_rejected > 0:
                logger.info(f"QC filter: masking {n_rejected} obs for {forecast_var} (QC > {max_qc_value})")
            obs_df.loc[bad, obs_col] = np.nan
    return obs_df


def convert_obs_units(obs_df, variable_map):
    """Apply unit conversions to observation columns as specified in variable map."""
    for forecast_var, vinfo in variable_map.items():
        conversion = vinfo["unit_conversion"]
        if conversion is not None:
            obs_col = vinfo["obs_col"]
            if obs_col in obs_df.columns:
                obs_df[obs_col] = UNIT_CONVERSIONS[conversion](obs_df[obs_col])
    return obs_df


def align_obs_to_forecast_times(obs_df, forecast_valid_times, window):
    """Match observations to forecast valid times within a temporal window.

    Args:
        obs_df: DataFrame with OBS_TIMESTAMP column.
        forecast_valid_times: Array of forecast valid times (np.datetime64).
        window: pd.Timedelta for +/- matching window.

    Returns:
        dict mapping pd.Timestamp -> DataFrame of matched observations.
    """
    aligned = {}
    obs_times = obs_df["OBS_TIMESTAMP"]

    # Remove timezone info if present, to match forecast times (tz-naive)
    if hasattr(obs_times.dtype, "tz") and obs_times.dtype.tz is not None:
        obs_times = obs_times.dt.tz_localize(None)

    for vt in forecast_valid_times:
        vtimestamp = pd.Timestamp(vt)
        mask = (obs_times >= vtimestamp - window) & (obs_times < vtimestamp + window)
        matched = obs_df.loc[mask]
        if len(matched) > 0:
            aligned[vtimestamp] = matched
    return aligned


def compute_obs_metrics(forecast_values, obs_values):
    """Compute verification metrics between forecast and observation values.

    Args:
        forecast_values: Array of forecast values at obs locations.
        obs_values: Array of observation values.

    Returns:
        dict with rmse, mae, bias, count.
    """
    valid = np.isfinite(forecast_values) & np.isfinite(obs_values)
    n = valid.sum()
    if n == 0:
        result = {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "count": 0}
    else:
        f = forecast_values[valid]
        o = obs_values[valid]
        diff = f - o
        result = {
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "mae": float(np.mean(np.abs(diff))),
            "bias": float(np.mean(diff)),
            "count": int(n),
        }
    xds = xr.Dataset(result)
    xds = xds.expand_dims({"time": [forecast_values["time"].values]})
#    if "level" in forecast_values.coords:
#        xds = xds.expand_dims({"level": [forecast_values["level"].values]})
    return xds


def main(config):
    """Verify forecasts against observations.

    See ``eagle-tools obs-metrics --help`` or cli.py for help.
    """

    topo = config["topo"]

    # Build variable map
    variable_map = build_variable_map(config)
    forecast_var_names = list(variable_map.keys())
    # Extract unique base variable names and levels for loading the forecast
    base_var_names = list({vinfo["base_name"] for vinfo in variable_map.values()})
    levels = sorted({vinfo["level"] for vinfo in variable_map.values()})
    logger.info(f"Variables to verify: {forecast_var_names}")

    # Config options
    model_type = config.get("model_type")
    lam_index = config.get("lam_index", None)
    lead_time = config["lead_time"]
    obs_dataset = config.get("obs_dataset", "conv-adpupa-NC002001")
    temporal_window = pd.Timedelta(config.get("temporal_window", "3h"))
    max_qc_value = config.get("max_qc_value", 2)

    # does the user want to evaluate on a different grid?
    # this doesn't include regridding the nested -> global resolution
    target_regrid_kwargs = config.get("target_regrid_kwargs", None)
    forecast_regrid_kwargs = config.get("forecast_regrid_kwargs", None)
    do_any_regridding = (target_regrid_kwargs is not None) or \
            ((forecast_regrid_kwargs is not None) and (model_type != "nested-global"))
    if do_any_regridding:
        raise NotImplementedError

    if model_type == "nested-global":
        forecast_regrid_kwargs["target_grid_path"] = prepare_regrid_target_mask(
            anemoi_reference_dataset_kwargs=config["anemoi_reference_dataset_kwargs"],
            horizontal_regrid_kwargs=forecast_regrid_kwargs,
        )

    # Generate initialization dates
    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])
    n_dates = len(dates)
    n_batches = int(np.ceil(n_dates / topo.size))

    container = {"rmse": [], "mae": [], "bias": [], "count": []}

    logger.info(f"Observation Verification")
    logger.info(f"Dataset: {obs_dataset}")
    logger.info(f"Temporal window: +/- {temporal_window}")
    logger.info(f"Max QC value: {max_qc_value}")
    logger.info(f"Initial Conditions:\n{dates}")

    for batch_idx in range(n_batches):

        date_idx = (batch_idx * topo.size) + topo.rank
        if date_idx + 1 > n_dates:
            break

        try:
            t0 = dates[date_idx]
        except Exception:
            logger.error(f"Error getting this date: {date_idx} / {n_dates}")
            raise

        st0 = t0.strftime("%Y-%m-%dT%H")
        logger.info(f"Processing {st0}")

        # Load forecast using base variable names and levels
        if config.get("from_anemoi", True):
            fname = f"{config['forecast_path']}/{st0}.{lead_time}h.nc"
            fds = open_anemoi_inference_dataset(
                fname,
                model_type=model_type,
                lam_index=lam_index,
                trim_edge=config.get("trim_forecast_edge", None),
                vars_of_interest=base_var_names,
                levels=levels,
                load=True,
                lcc_info=config.get("lcc_info", None),
                horizontal_regrid_kwargs=forecast_regrid_kwargs if model_type == "nested-global" else None,
                reshape_cell_to_2d=True,
            )
        else:
            fds = open_forecast_zarr_dataset(
                config["forecast_path"],
                t0=t0,
                vars_of_interest=base_var_names,
                levels=levels,
                load=True,
                lcc_info=config.get("lcc_info", None),
            )

        # Pad in longitude
        if "global" in model_type:
            dlon = fds["longitude"].diff("longitude").values[0]
            fds = fds.pad({"longitude": 1}, mode="wrap")
            plon = np.concatenate([
                [fds["longitude"][1].values - dlon],
                fds["longitude"].values[1:-1],
                [fds["longitude"][-2].values + dlon],
            ])
            fds["longitude"] = plon

        # Get forecast valid times
        logger.info(f"Opened fds\n{fds}")
        forecast_valid_times = fds["time"].values
        logger.info(f"vtimes: {forecast_valid_times}")

        # Load observations for the full valid time range (padded by window)
        time_start = pd.Timestamp(forecast_valid_times[0]) - temporal_window
        time_end = pd.Timestamp(forecast_valid_times[-1]) + temporal_window
        obs_df = load_observations(config, (time_start, time_end), variable_map)

        # QC filter and unit conversion
        obs_df = apply_qc_filter(obs_df, variable_map, max_qc_value=max_qc_value)
        obs_df = convert_obs_units(obs_df, variable_map)

        # Convert obs longitudes to 0-360 to match forecast convention
        obs_df["LON"] = obs_df["LON"] % 360

        # Align observations to forecast valid times
        aligned = align_obs_to_forecast_times(obs_df, forecast_valid_times, temporal_window)

        # Compute metrics per forecast valid time
        container_per_ic = {metric: {varname: [] for varname in forecast_var_names} for metric in container.keys()}

        for vtime in forecast_valid_times:
            vtimestamp = pd.Timestamp(vtime)

            # Handle case where we don't have obs
            if vtimestamp not in aligned:
                for varname in forecast_var_names:
                    for metric in container.keys():
                        fillvalue = 0 if metric == "count" else np.nan
                        fillds = xr.DataArray(fillvalue, coords={"time": [vtime]}, dims=("time",))
                        container_per_ic[metric][varname].append(fillds)
                continue

            # TODO: maybe do this conversion earlier?
            matched_obs = aligned[vtimestamp].to_xarray()

            # Interp to obs locations and compute metrics
            interpolated = fds.sel(time=vtime).interp({
                "longitude": matched_obs["LON"],
                "latitude": matched_obs["LAT"],
            })
            for varname, vinfo in variable_map.items():
                fvals = interpolated[vinfo["base_name"]]
                if "level" in fvals.dims:
                    fvals = fvals.sel(level=vinfo["level"])

                result = compute_obs_metrics(fvals, matched_obs[vinfo["obs_col"]])
                for metric in container.keys():
                    container_per_ic[metric][varname].append(result[metric])

        # Assemble into xr.Dataset
        for metric, thedata in container_per_ic.items():
            data_vars = {}
            for varname, vals in thedata.items():
                data_vars[varname] = xr.concat(vals, dim="time")

            this_metric_ds = postprocess(xr.Dataset(data_vars))
            container[metric].append(this_metric_ds)

        logger.info(f"Done with {st0}")

    logger.info("Done Computing Observation Verification Metrics")

    logger.info("Gathering Results on Root Process")
    for name in container.keys():
        container[name] = topo.gather(container[name])

    if topo.is_root:
        for name in container:
            c = container[name]
            if config["use_mpi"]:
                c = [xds for sublist in c for xds in sublist]
            c = sorted(c, key=lambda xds: xds.coords["t0"])
            container[name] = xr.concat(c, dim="t0")

        for metric, xds in container.items():
            fname = f"{config['output_path']}/obs.{metric}.{obs_dataset}.nc"
            xds.to_netcdf(fname)
            logger.info(f"Stored result: {fname}")
        logger.info("Done Storing Observation Verification Metrics")
