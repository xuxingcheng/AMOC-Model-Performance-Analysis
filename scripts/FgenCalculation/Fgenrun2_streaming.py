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


def validate_files(files):
    clean_files = []
    for path in files:
        if os.path.getsize(path) == 0:
            print(f"  Skipping empty file: {path}")
            continue
        try:
            xr.open_dataset(path, engine="netcdf4").close()
            clean_files.append(path)
        except Exception as exc:
            print(f"  Skipping unreadable file: {path}")
            print(f"    Error: {exc!r}")
    return clean_files


def open_mfdataset_once(model, variable_key, files, time_chunk):
    open_kwargs = {
        "combine": "by_coords",
        "coords": "minimal",
        "data_vars": "minimal",
        "compat": "override",
        "join": "override",
        "parallel": False,
        "engine": "netcdf4",
        "use_cftime": True,
        "chunks": {"time": time_chunk},
    }

    try:
        return xr.open_mfdataset(files, **open_kwargs)
    except Exception as exc:
        print(
            f"{model} {variable_key}: open_mfdataset failed once; "
            "retrying after filtering bad files."
        )
        print(f"  First error: {exc!r}")
        clean_files = validate_files(files)
        if not clean_files:
            raise RuntimeError(f"{model} {variable_key}: no valid files remain") from exc
        return xr.open_mfdataset(clean_files, **open_kwargs)


def open_model_inputs(model, registry, time_chunk):
    ds_map = {}
    for variable_key in VARIABLE_SPECS:
        files = registry[model][variable_key]
        ds_map[variable_key] = open_mfdataset_once(
            model=model,
            variable_key=variable_key,
            files=files,
            time_chunk=time_chunk,
        )
    return ds_map


def align_and_trim_inputs(model, opened_ds_map, last_n_years, last_n_months):
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
        trimmed[variable_key] = selected

    return trimmed


def compute_surface_density(trimmed_ds_map):
    T, SP = xr.align(
        trimmed_ds_map["tos"]["tos"],
        trimmed_ds_map["sos"]["sos"],
        join="inner",
    )
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


def compute_fsurf(trimmed_ds_map):
    cp = 3990.0
    rho0 = 1027.0
    rho_fw = 1000.0
    S0 = 35.0

    HF, WF = xr.align(
        trimmed_ds_map["hfds"]["hfds"],
        trimmed_ds_map["wfo"]["wfo"],
        join="inner",
    )
    rho, alpha, beta = compute_surface_density(trimmed_ds_map)
    HF, WF, rho, alpha, beta = xr.align(HF, WF, rho, alpha, beta, join="inner")

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
        ds = xr.open_dataset(area_index[source_model][0], engine="netcdf4")
        area = ds["areacello"]
        if "time" in area.dims:
            area = area.isel(time=0).squeeze()
        else:
            area = area.squeeze()
        return area.load()
    finally:
        safe_close(ds)


def get_spatial_dims(fsurf_ds):
    dims = tuple(dim for dim in fsurf_ds["fsurf"].dims if dim != "time")

    if dims == ("i",):
        return dims
    if set(dims) == {"j", "i"}:
        return ("j", "i")
    if set(dims) == {"y", "x"}:
        return ("y", "x")
    if set(dims) == {"lat", "lon"}:
        return ("lat", "lon")
    return None


def get_latitude_mask(fsurf_ds, spatial_dims):
    sample = fsurf_ds["fsurf"].isel(time=0)

    if "latitude" in fsurf_ds:
        latitude = fsurf_ds["latitude"]
        if set(latitude.dims) != set(spatial_dims):
            latitude, _ = xr.broadcast(latitude, sample)
        return latitude > 45

    if "lat" in fsurf_ds.coords:
        latitude = fsurf_ds["lat"]
        mask = latitude > 45
        if set(mask.dims) != set(spatial_dims):
            mask, _ = xr.broadcast(mask, sample)
        return mask

    raise KeyError("No latitude or lat coordinate available for filtering.")


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


def stack_north_of_45(fsurf_ds, area_da, spatial_dims):
    mask = get_latitude_mask(fsurf_ds, spatial_dims)
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

    area_ready = prepare_area_for_fsurf(area_da, fsurf_ds, spatial_dims)
    stacked = stack_north_of_45(fsurf_ds, area_ready, spatial_dims)
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
        trimmed_ds_map = None
        area_da = None
        fsurf_ds = None

        try:
            opened_ds_map = open_model_inputs(model, registry, time_chunk=time_chunk)
            trimmed_ds_map = align_and_trim_inputs(
                model=model,
                opened_ds_map=opened_ds_map,
                last_n_years=args.last_n_years,
                last_n_months=args.last_n_months,
            )
            fsurf_ds = compute_fsurf(trimmed_ds_map)

            area_da = load_area_for_model(model, area_index)
            if area_da is None:
                print(f"Skipping {model}: no areacello file or alias.")
                continue

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
            del opened_ds_map, trimmed_ds_map, area_da, fsurf_ds
            gc.collect()

    if unsaved:
        save_results(args.output, results)
        print(f"Final checkpoint saved with {len(results)} models")

    print(f"Done. Total models in output: {len(results)}")


if __name__ == "__main__":
    with dask.config.set({"array.slicing.split_large_chunks": True}):
        main()
