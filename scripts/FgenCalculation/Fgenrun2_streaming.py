import argparse
import gc
import glob
import os
import pickle
import warnings
from collections import defaultdict

import dask
import gsw
import numpy as np
import pandas as pd
import xarray as xr


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Not a valid ID")

DATA_ROOT = "/glade/work/stevenxu/AMOC_models"
AREA_DIR = os.path.join(DATA_ROOT, "areacello")
DEFAULT_OUTPUT = os.path.join(DATA_ROOT, "Fgen_Allmodels_streaming.pkl")
DEFAULT_SCENARIO = "PIControl"
DEFAULT_LAST_N_YEARS = 20
DEFAULT_LAST_N_MONTHS = 240
DEFAULT_STEP_SIZE = 0.05
DEFAULT_RHO_MIN = 1015.0
DEFAULT_RHO_MAX = 1030.0
DEFAULT_EXCLUDING_MODELS = {"CESM2"}
ENGINE_FALLBACK_ORDER = ("h5netcdf", "scipy", "netcdf4")
AREA_ALIASES = {
    "FGOALS-f3-L": "ACCESS-CM2",
    "FGOALS-g3": "ACCESS-CM2",
}
VARIABLE_SPECS = {
    "tos": {
        "directory": "sea_surface_temperature",
        "variable_name": "tos",
    },
    "sos": {
        "directory": "sea_surface_salinity",
        "variable_name": "sos",
    },
    "hfds": {
        "directory": "heatflux",
        "variable_name": "hfds",
    },
    "wfo": {
        "directory": "waterflux",
        "variable_name": "wfo",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sequential Fgen driver that processes one model at a time to avoid "
            "the I/O blow-up from launching many independent jobs."
        )
    )
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--last-n-years", type=int, default=DEFAULT_LAST_N_YEARS)
    parser.add_argument("--last-n-months", type=int, default=DEFAULT_LAST_N_MONTHS)
    parser.add_argument("--rho-min", type=float, default=DEFAULT_RHO_MIN)
    parser.add_argument("--rho-max", type=float, default=DEFAULT_RHO_MAX)
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--time-chunk", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=sorted(DEFAULT_EXCLUDING_MODELS),
    )
    return parser.parse_args()


def subtract_years_cftime(t, years):
    return type(t)(
        t.year - years,
        t.month,
        t.day,
        t.hour,
        t.minute,
        t.second,
        t.microsecond,
        has_year_zero=getattr(t, "has_year_zero", False),
    )


def safe_close(obj):
    close = getattr(obj, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def group_files_by_model(directory):
    groups = defaultdict(list)
    for fp in glob.glob(os.path.join(directory, "*.nc")):
        fname = os.path.basename(fp)
        parts = fname.split("_")
        if len(parts) < 3:
            continue
        groups[parts[2]].append(fp)

    return {model: sorted(files) for model, files in groups.items()}


def build_model_file_registry(scenario):
    registry = defaultdict(dict)
    for variable_key, spec in VARIABLE_SPECS.items():
        directory = os.path.join(DATA_ROOT, spec["directory"], "scenarios", scenario)
        grouped = group_files_by_model(directory)
        for model, files in grouped.items():
            registry[model][variable_key] = files
    return registry


def get_candidate_models(registry, include_models=None, exclude_models=None):
    exclude = set(exclude_models or [])
    include = set(include_models) if include_models else None

    models = []
    for model, file_map in registry.items():
        if any(variable_key not in file_map for variable_key in VARIABLE_SPECS):
            continue
        if model in exclude:
            continue
        if include is not None and model not in include:
            continue
        models.append(model)

    return sorted(models)


def is_classic_netcdf_error(exc):
    message = str(exc).lower()
    needles = (
        "file signature not found",
        "not a valid netcdf 4 file",
        "not a netcdf 4 file",
        "unknown file format",
        "not a valid netcdf file",
    )
    return any(needle in message for needle in needles)


def summarize_engine_errors(errors):
    parts = []
    for engine, exc in errors.items():
        parts.append(f"{engine}: {exc!r}")
    return "; ".join(parts)


def build_mfdataset_kwargs(engine, time_chunk):
    return {
        "combine": "by_coords",
        "coords": "minimal",
        "data_vars": "minimal",
        "compat": "override",
        "join": "override",
        "parallel": False,
        "engine": engine,
        "use_cftime": True,
        "chunks": {"time": time_chunk},
    }


def build_dataset_kwargs(engine):
    return {
        "engine": engine,
        "use_cftime": True,
    }


def validate_files(files):
    clean_files = []
    for path in files:
        if os.path.getsize(path) == 0:
            print(f"  Skipping empty file: {path}")
            continue
        ds = None
        try:
            ds, _ = open_dataset_with_fallback(path, label=path, log_backend=False)
            clean_files.append(path)
        except Exception as exc:
            print(f"  Skipping unreadable file: {path}")
            print(f"    Error: {exc!r}")
        finally:
            safe_close(ds)
    return clean_files


def open_dataset_with_fallback(path, label, log_backend=True):
    errors = {}
    for engine in ENGINE_FALLBACK_ORDER:
        try:
            ds = xr.open_dataset(path, **build_dataset_kwargs(engine))
            if log_backend:
                print(f"{label}: using backend {engine}")
            return ds, engine
        except Exception as exc:
            errors[engine] = exc

    raise RuntimeError(
        f"{label}: unable to open dataset with fallback backends: "
        f"{summarize_engine_errors(errors)}"
    )


def open_mfdataset_once(model, variable_key, files, time_chunk):
    errors = {}
    for engine in ENGINE_FALLBACK_ORDER:
        try:
            ds = xr.open_mfdataset(files, **build_mfdataset_kwargs(engine, time_chunk))
            print(f"{model} {variable_key}: using backend {engine}")
            return ds, engine
        except Exception as exc:
            errors[engine] = exc

    print(
        f"{model} {variable_key}: all backends failed on first pass; "
        "retrying after filtering bad files."
    )
    print(f"  Backend errors: {summarize_engine_errors(errors)}")
    clean_files = validate_files(files)
    if not clean_files:
        raise RuntimeError(f"{model} {variable_key}: no valid files remain")

    retry_errors = {}
    for engine in ENGINE_FALLBACK_ORDER:
        try:
            ds = xr.open_mfdataset(clean_files, **build_mfdataset_kwargs(engine, time_chunk))
            print(f"{model} {variable_key}: using backend {engine} after file filtering")
            return ds, engine
        except Exception as exc:
            retry_errors[engine] = exc

    raise RuntimeError(
        f"{model} {variable_key}: unable to open dataset with fallback backends after "
        f"file filtering: {summarize_engine_errors(retry_errors)}"
    )


def open_model_inputs(model, registry, time_chunk):
    ds_map = {}
    backend_map = {}
    for variable_key in VARIABLE_SPECS:
        files = registry[model][variable_key]
        ds_map[variable_key], backend_map[variable_key] = open_mfdataset_once(
            model=model,
            variable_key=variable_key,
            files=files,
            time_chunk=time_chunk,
        )
    return ds_map, backend_map


def align_and_trim_inputs(model, opened_ds_map, backend_map, last_n_years, last_n_months):
    end_times = [
        opened_ds_map["tos"]["time"].isel(time=-1).values.item(),
        opened_ds_map["sos"]["time"].isel(time=-1).values.item(),
        opened_ds_map["hfds"]["time"].isel(time=-1).values.item(),
        opened_ds_map["wfo"]["time"].isel(time=-1).values.item(),
    ]

    min_end_time = min(end_times)
    start_time = subtract_years_cftime(min_end_time, last_n_years)

    print(
        f"Aligning {model} to {start_time} -> {min_end_time} "
        f"and keeping the last {last_n_months} months."
    )

    trimmed = {}
    for variable_key, ds in opened_ds_map.items():
        selected = ds.sel(time=slice(start_time, min_end_time))
        if last_n_months is not None:
            selected = selected.isel(time=slice(-last_n_months, None))
        if backend_map.get(variable_key) == "scipy":
            selected = selected.load()
        trimmed[variable_key] = selected

    return trimmed


def get_dataarray_spatial_dims(data_array):
    return tuple(dim for dim in data_array.dims if dim != "time")


def assert_nonempty_spatial_grid(model, data_array, label):
    spatial_dims = get_dataarray_spatial_dims(data_array)
    if not spatial_dims:
        raise ValueError(f"{model}: {label} has no spatial dimensions")

    empty_dims = [dim for dim in spatial_dims if data_array.sizes.get(dim, 0) == 0]
    if empty_dims:
        dim_text = ", ".join(empty_dims)
        raise ValueError(
            f"{model}: empty spatial grid after harmonization/alignment for {label} "
            f"({dim_text})"
        )


def assert_matching_spatial_grid(model, reference, other, reference_name, other_name):
    reference_dims = get_dataarray_spatial_dims(reference)
    other_dims = get_dataarray_spatial_dims(other)
    if reference_dims != other_dims:
        raise ValueError(
            f"{model}: grid mismatch after harmonization: {other_name} dims "
            f"{other_dims} do not match {reference_name} dims {reference_dims}"
        )

    for dim in reference_dims:
        if dim in reference.coords and dim in other.coords:
            reference_values = reference[dim].values
            other_values = other[dim].values
            if reference_values.shape != other_values.shape or not np.array_equal(
                reference_values,
                other_values,
            ):
                raise ValueError(
                    f"{model}: grid mismatch after harmonization: {other_name} "
                    f"coordinate '{dim}' does not match {reference_name}"
                )


def area_weighted_coarsen_to_target(model, variable_name, data_array, area_da, target_array):
    if get_dataarray_spatial_dims(data_array) != ("lat", "lon"):
        raise ValueError(
            f"{model}: cannot coarsen {variable_name}; expected (lat, lon) grid, "
            f"got {data_array.dims}"
        )

    if area_da.dims != ("lat", "lon"):
        raise ValueError(
            f"{model}: cannot coarsen {variable_name}; areacello dims {area_da.dims} "
            "do not match expected (lat, lon)"
        )

    if (
        area_da.sizes.get("lat") != data_array.sizes.get("lat")
        or area_da.sizes.get("lon") != data_array.sizes.get("lon")
    ):
        raise ValueError(
            f"{model}: cannot coarsen {variable_name}; areacello shape "
            f"{area_da.shape} does not match {variable_name} shape "
            f"{data_array.shape[1:]}"
        )

    if not np.array_equal(area_da["lat"].values, data_array["lat"].values):
        area_da = area_da.assign_coords(lat=data_array["lat"])
    if not np.array_equal(area_da["lon"].values, data_array["lon"].values):
        area_da = area_da.assign_coords(lon=data_array["lon"])

    area_sum = area_da.coarsen(lat=2, lon=2, boundary="trim").sum()
    weighted_sum = (data_array * area_da).coarsen(lat=2, lon=2, boundary="trim").sum()
    coarse = xr.where(area_sum > 0, weighted_sum / area_sum, np.nan)
    coarse = coarse.assign_coords(lat=target_array["lat"], lon=target_array["lon"])
    coarse = coarse.transpose(*target_array.dims)
    coarse.attrs = data_array.attrs.copy()

    print(
        f"{model} {variable_name}: coarsened from "
        f"{data_array.sizes['lat']}x{data_array.sizes['lon']} to "
        f"{target_array.sizes['lat']}x{target_array.sizes['lon']} using area-weighted averaging"
    )
    return coarse


def harmonize_spatial_grids(model, trimmed_ds_map, area_da):
    harmonized = dict(trimmed_ds_map)
    target = harmonized["tos"]["tos"]
    assert_nonempty_spatial_grid(model, target, "tos")

    for variable_key in ("sos", "hfds"):
        assert_nonempty_spatial_grid(model, harmonized[variable_key][variable_key], variable_key)
        assert_matching_spatial_grid(
            model,
            target,
            harmonized[variable_key][variable_key],
            "tos",
            variable_key,
        )

    wfo = harmonized["wfo"]["wfo"]
    assert_nonempty_spatial_grid(model, wfo, "wfo")

    target_dims = get_dataarray_spatial_dims(target)
    wfo_dims = get_dataarray_spatial_dims(wfo)
    if target_dims == ("lat", "lon") and wfo_dims == ("lat", "lon"):
        is_two_x_finer = (
            wfo.sizes.get("lat") == target.sizes.get("lat") * 2
            and wfo.sizes.get("lon") == target.sizes.get("lon") * 2
        )
        if is_two_x_finer:
            coarse_wfo = area_weighted_coarsen_to_target(
                model=model,
                variable_name="wfo",
                data_array=wfo,
                area_da=area_da,
                target_array=target,
            )
            harmonized["wfo"] = coarse_wfo.to_dataset(name="wfo")
            wfo = harmonized["wfo"]["wfo"]

    assert_matching_spatial_grid(model, target, wfo, "tos", "wfo")
    return harmonized


def compute_surface_density(model, trimmed_ds_map):
    T, SP = xr.align(
        trimmed_ds_map["tos"]["tos"],
        trimmed_ds_map["sos"]["sos"],
        join="exact",
    )
    assert_nonempty_spatial_grid(model, T, "tos/sos after align")
    p0 = 0.0

    alpha = xr.apply_ufunc(
        gsw.density.alpha,
        SP,
        T,
        p0,
        dask="parallelized",
        output_dtypes=[float],
    ).rename("alpha")

    beta = xr.apply_ufunc(
        gsw.density.beta,
        SP,
        T,
        p0,
        dask="parallelized",
        output_dtypes=[float],
    ).rename("beta")

    rho = xr.apply_ufunc(
        gsw.density.rho,
        SP,
        T,
        p0,
        dask="parallelized",
        output_dtypes=[float],
    ).rename("rho")

    rho = rho.assign_attrs(units="kg m-3", long_name="Sea-surface density")
    return rho, alpha, beta


def compute_fsurf(model, trimmed_ds_map, area_da):
    cp = 3990.0
    rho0 = 1027.0
    rho_fw = 1000.0
    S0 = 35.0

    trimmed_ds_map = harmonize_spatial_grids(model, trimmed_ds_map, area_da)
    HF, WF = xr.align(
        trimmed_ds_map["hfds"]["hfds"],
        trimmed_ds_map["wfo"]["wfo"],
        join="exact",
    )
    assert_nonempty_spatial_grid(model, HF, "hfds/wfo after align")
    rho, alpha, beta = compute_surface_density(model, trimmed_ds_map)
    HF, WF, rho, alpha, beta = xr.align(HF, WF, rho, alpha, beta, join="exact")
    assert_nonempty_spatial_grid(model, HF, "final fsurf inputs")

    fsurf = (alpha / cp) * HF + (rho0 / rho_fw) * beta * S0 * WF
    fsurf = fsurf.rename("fsurf").assign_attrs(
        long_name="Buoyancy-relevant surface forcing (Eq. 5)",
        description="(alpha/cp)*f_heat + (rho0/rho_fw)*beta*S0*f_water",
        cp=cp,
        rho0=rho0,
        rho_fw=rho_fw,
        S0=S0,
    )
    heat_comp = ((alpha / cp) * HF).rename("heat_comp")
    fw_comp = ((rho0 / rho_fw) * beta * S0 * WF).rename("fw_comp")

    return xr.Dataset(
        data_vars={
            "fsurf": fsurf,
            "rho": rho,
            "heat_comp": heat_comp,
            "fw_comp": fw_comp,
        }
    )


def load_area_for_model(model, area_index):
    source_model = model
    if source_model not in area_index:
        source_model = AREA_ALIASES.get(model)
        if source_model is None or source_model not in area_index:
            return None
        print(f"Using areacello from {source_model} for {model}")

    ds = None
    try:
        label = f"{model} areacello"
        ds, _ = open_dataset_with_fallback(area_index[source_model][0], label=label)
        area = ds["areacello"]
        if "time" in area.dims:
            area = area.isel(time=0).squeeze()
        else:
            area = area.squeeze()
        return area.load()
    finally:
        safe_close(ds)


def get_spatial_dims(fsurf_ds):
    dims = get_dataarray_spatial_dims(fsurf_ds["fsurf"])

    if dims == ("i",):
        return dims
    if set(dims) == {"j", "i"}:
        return ("j", "i")
    if set(dims) == {"y", "x"}:
        return ("y", "x")
    if set(dims) == {"lat", "lon"}:
        return ("lat", "lon")
    return None


def normalize_longitude(longitude):
    return ((longitude + 180) % 360) - 180


def get_spatial_mask(fsurf_ds, spatial_dims):
    sample = fsurf_ds["fsurf"].isel(time=0)

    if "latitude" in fsurf_ds:
        latitude = fsurf_ds["latitude"]
        if "longitude" not in fsurf_ds:
            raise KeyError("No longitude field available for filtering.")
        longitude = fsurf_ds["longitude"]
        if set(latitude.dims) != set(spatial_dims):
            latitude, _ = xr.broadcast(latitude, sample)
        if set(longitude.dims) != set(spatial_dims):
            longitude, _ = xr.broadcast(longitude, sample)
        longitude = normalize_longitude(longitude)
        return (latitude > 45) & (longitude >= -90) & (longitude <= 60)

    if "lat" in fsurf_ds.coords:
        latitude = fsurf_ds["lat"]
        if "lon" not in fsurf_ds.coords:
            raise KeyError("No lon coordinate available for filtering.")
        longitude = normalize_longitude(fsurf_ds["lon"])
        mask = (latitude > 45) & (longitude >= -90) & (longitude <= 60)
        if set(mask.dims) != set(spatial_dims):
            mask, _ = xr.broadcast(mask, sample)
        return mask

    raise KeyError("No latitude/longitude coordinates available for filtering.")


def prepare_area_for_fsurf(area_da, fsurf_ds, spatial_dims):
    if spatial_dims == ("lat", "lon"):
        if (
            area_da.sizes.get("lat", 0) == fsurf_ds.sizes.get("lat", 0) * 2
            and area_da.sizes.get("lon", 0) == fsurf_ds.sizes.get("lon", 0) * 2
        ):
            area_da = area_da.coarsen(lat=2, lon=2, boundary="trim").sum()
        return area_da.sel(
            lat=fsurf_ds["lat"],
            lon=fsurf_ds["lon"],
            method="nearest",
        )

    indexers = {}
    for dim in spatial_dims:
        if dim in area_da.dims and dim in fsurf_ds.coords:
            indexers[dim] = fsurf_ds[dim]
    if indexers:
        area_da = area_da.sel(indexers)
    return area_da


def stack_north_of_45(model, fsurf_ds, area_da, spatial_dims):
    assert_nonempty_spatial_grid(model, fsurf_ds["fsurf"], "fsurf for latitude mask")
    mask = get_spatial_mask(fsurf_ds, spatial_dims)
    mask_s = mask.stack(points=spatial_dims)
    keep_pts = mask_s.where(mask_s, drop=True).coords["points"]
    if keep_pts.size == 0:
        return None

    rho_s = fsurf_ds["rho"].stack(points=spatial_dims).sel(points=keep_pts)
    fsurf_s = fsurf_ds["fsurf"].stack(points=spatial_dims).sel(points=keep_pts)
    heat_s = fsurf_ds["heat_comp"].stack(points=spatial_dims).sel(points=keep_pts)
    fw_s = fsurf_ds["fw_comp"].stack(points=spatial_dims).sel(points=keep_pts)
    area_s = area_da.stack(points=spatial_dims).sel(points=keep_pts)

    return rho_s, fsurf_s, heat_s, fw_s, area_s


def aggregate_density_bins(rho_np, wf_np, wh_np, wfw_np, area_np, rho_classes, step_size):
    n_time = rho_np.shape[0]
    n_bins = len(rho_classes)
    lower = rho_classes[0]
    upper = rho_classes[-1] + step_size
    rho_centers = rho_classes + step_size / 2.0
    area_flat = np.ravel(area_np)

    rows = []
    for t in range(n_time):
        rho_flat = np.ravel(rho_np[t])
        wf_flat = np.ravel(wf_np[t])
        wh_flat = np.ravel(wh_np[t])
        wfw_flat = np.ravel(wfw_np[t])

        valid = np.isfinite(rho_flat) & (rho_flat > lower) & (rho_flat < upper)
        if np.any(valid):
            bin_idx = np.floor((rho_flat[valid] - lower) / step_size).astype(np.int64)
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)

            fgen_sum = np.bincount(
                bin_idx,
                weights=np.nan_to_num(wf_flat[valid], nan=0.0),
                minlength=n_bins,
            ).astype(float)
            heat_sum = np.bincount(
                bin_idx,
                weights=np.nan_to_num(wh_flat[valid], nan=0.0),
                minlength=n_bins,
            ).astype(float)
            fw_sum = np.bincount(
                bin_idx,
                weights=np.nan_to_num(wfw_flat[valid], nan=0.0),
                minlength=n_bins,
            ).astype(float)
            area_sum = np.bincount(
                bin_idx,
                weights=np.nan_to_num(area_flat[valid], nan=0.0),
                minlength=n_bins,
            ).astype(float)
        else:
            fgen_sum = np.zeros(n_bins, dtype=float)
            heat_sum = np.zeros(n_bins, dtype=float)
            fw_sum = np.zeros(n_bins, dtype=float)
            area_sum = np.zeros(n_bins, dtype=float)

        for idx, rho_center in enumerate(rho_centers):
            rows.append(
                [
                    t,
                    rho_center,
                    fgen_sum[idx] / step_size / 1e6,
                    heat_sum[idx] / step_size / 1e6,
                    fw_sum[idx] / step_size / 1e6,
                    area_sum[idx],
                ]
            )

    df = pd.DataFrame(
        rows,
        columns=["time", "rho", "Fgen", "HeatFlux", "FreshwaterFlux", "AreaSum"],
    )
    return df.groupby("rho", as_index=False)[
        ["Fgen", "HeatFlux", "FreshwaterFlux", "AreaSum"]
    ].mean()


def compute_fgen_for_model(model, fsurf_ds, area_da, rho_classes, step_size):
    spatial_dims = get_spatial_dims(fsurf_ds)
    if spatial_dims is None:
        print(f"Skipping {model}: unsupported spatial dims {fsurf_ds['fsurf'].dims}")
        return None

    assert_nonempty_spatial_grid(model, fsurf_ds["fsurf"], "fsurf before integration")
    area_ready = prepare_area_for_fsurf(area_da, fsurf_ds, spatial_dims)
    stacked = stack_north_of_45(model, fsurf_ds, area_ready, spatial_dims)
    if stacked is None:
        print(f"Skipping {model}: no points north of 45 degrees.")
        return None

    rho_s, fsurf_s, heat_s, fw_s, area_s = stacked
    weighted_fsurf = fsurf_s * area_s
    weighted_heat = heat_s * area_s
    weighted_fw = fw_s * area_s

    rho_np, wf_np, wh_np, wfw_np, area_np = dask.compute(
        rho_s.data,
        weighted_fsurf.data,
        weighted_heat.data,
        weighted_fw.data,
        area_s.data,
    )

    return aggregate_density_bins(
        rho_np=rho_np,
        wf_np=wf_np,
        wh_np=wh_np,
        wfw_np=wfw_np,
        area_np=area_np,
        rho_classes=rho_classes,
        step_size=step_size,
    )


def save_results(output_path, results):
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(results, f)
    os.replace(tmp_path, output_path)


def load_existing_results(output_path):
    if not os.path.exists(output_path):
        return {}
    with open(output_path, "rb") as f:
        return pickle.load(f)


def main():
    args = parse_args()
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")

    if args.time_chunk is None:
        if args.last_n_months is not None:
            time_chunk = max(12, args.last_n_months)
        else:
            time_chunk = max(12, args.last_n_years * 12)
    else:
        time_chunk = args.time_chunk

    rho_classes = np.arange(
        args.rho_min - args.step_size,
        args.rho_max + args.step_size,
        args.step_size,
    )

    registry = build_model_file_registry(args.scenario)
    area_index = group_files_by_model(AREA_DIR)
    candidate_models = get_candidate_models(
        registry=registry,
        include_models=args.models,
        exclude_models=args.exclude_models,
    )

    if args.max_models is not None:
        candidate_models = candidate_models[: args.max_models]

    results = load_existing_results(args.output) if args.resume else {}
    processed_before = set(results)
    models = [model for model in candidate_models if model not in processed_before]

    print(f"Scenario: {args.scenario}")
    print(f"Candidate models with tos/sos/hfds/wfo: {len(candidate_models)}")
    print(f"Models remaining this run: {len(models)}")
    print(f"Output: {args.output}")

    unsaved = 0
    for model in models:
        print(f"\n=== Processing {model} ===")
        opened_ds_map = None
        backend_map = None
        trimmed_ds_map = None
        area_da = None
        fsurf_ds = None

        try:
            opened_ds_map, backend_map = open_model_inputs(
                model,
                registry,
                time_chunk=time_chunk,
            )
            trimmed_ds_map = align_and_trim_inputs(
                model=model,
                opened_ds_map=opened_ds_map,
                backend_map=backend_map,
                last_n_years=args.last_n_years,
                last_n_months=args.last_n_months,
            )
            area_da = load_area_for_model(model, area_index)
            if area_da is None:
                print(f"Skipping {model}: no areacello file or alias.")
                continue

            # Materialize one model at a time to avoid dask fancy-index bugs during
            # the later stack/select integration step on mixed backend inputs.
            fsurf_ds = compute_fsurf(model, trimmed_ds_map, area_da).load()
            fgen = compute_fgen_for_model(
                model=model,
                fsurf_ds=fsurf_ds,
                area_da=area_da,
                rho_classes=rho_classes,
                step_size=args.step_size,
            )
            if fgen is None or fgen.empty:
                print(f"Skipping {model}: no Fgen rows produced.")
                continue

            results[model] = fgen
            unsaved += 1
            print(f"Completed {model}: {len(fgen)} rho bins")

            if unsaved >= args.save_every:
                save_results(args.output, results)
                print(f"Checkpoint saved with {len(results)} models")
                unsaved = 0

        except Exception as exc:
            print(f"Error while processing {model}: {exc!r}")

        finally:
            if opened_ds_map is not None:
                for ds in opened_ds_map.values():
                    safe_close(ds)
            if trimmed_ds_map is not None:
                for ds in trimmed_ds_map.values():
                    safe_close(ds)
            safe_close(fsurf_ds)
            del opened_ds_map, backend_map, trimmed_ds_map, area_da, fsurf_ds
            gc.collect()

    if unsaved:
        save_results(args.output, results)
        print(f"Final checkpoint saved with {len(results)} models")

    print(f"Done. Total models in output: {len(results)}")


if __name__ == "__main__":
    with dask.config.set({"array.slicing.split_large_chunks": True}):
        main()
