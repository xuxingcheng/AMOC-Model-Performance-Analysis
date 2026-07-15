#!/usr/bin/env python3
"""Build a raw-field multi-model mean and calculate Fgen on the mean fields.

This is an intentionally separate experiment from the production native-grid
workflow.  Each model's raw ``tos``, ``sos``, ``hfds``, and ``wfo`` fields are
first aggregated to a shared regular grid with the source-area-weighted binned
method.  Per-model monthly climatologies are then averaged with equal model
weight before the existing Fgen helpers are applied to the four saved mean
files.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import dask
import numpy as np
import xarray as xr


SCRIPT_DIR = Path(__file__).resolve().parent
GRID_DIR = SCRIPT_DIR.parent / "GridAlignment"
for import_dir in (SCRIPT_DIR, GRID_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import Fgenrun2_streaming as fgen
import grids


DEFAULT_DATA_ROOT = Path(fgen.DATA_ROOT)
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "MMM_binned_1deg_no_SAM0"
DEFAULT_LEGACY_FGEN = DEFAULT_DATA_ROOT / "Fgen_Allmodels_streaming.pkl"
DEFAULT_MODELS = (
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "CAS-ESM2-0",
    "CanESM5",
    "CanESM5-1",
    "E3SM-1-0",
    "GISS-E2-1-G-CC",
    "GISS-E2-2-G",
    "ICON-ESM-LR",
    "MIROC6",
    "MPI-ESM-1-2-HAM",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "NorESM2-LM",
    "NorESM2-MM",
)
VARIABLE_KEYS = tuple(fgen.VARIABLE_SPECS)
MONTH_NUMBERS = np.arange(1, 13, dtype=np.int16)
CHECKPOINT_SCHEMA_VERSION = 2
RESULT_SCHEMA_VERSION = 1
ALGORITHM_VERSION = 2
AREA_RELATIVE_ERROR_TOLERANCE = 1.0e-6
GISS_WFO_AREA_COORDINATE_MODELS = {
    "GISS-E2-1-G-CC",
    "GISS-E2-2-G",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--legacy-fgen", type=Path, default=DEFAULT_LEGACY_FGEN)
    parser.add_argument("--scenario", default=fgen.DEFAULT_SCENARIO)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument(
        "--last-n-years",
        type=int,
        default=fgen.DEFAULT_LAST_N_YEARS,
        help=(
            "Compatibility/lookback guard retained from Fgenrun2_streaming.py; "
            "it must span at least --last-n-months."
        ),
    )
    parser.add_argument("--last-n-months", type=int, default=fgen.DEFAULT_LAST_N_MONTHS)
    parser.add_argument("--time-chunk", type=int, default=12)
    parser.add_argument("--rho-min", type=float, default=fgen.DEFAULT_RHO_MIN)
    parser.add_argument("--rho-max", type=float, default=fgen.DEFAULT_RHO_MAX)
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.05,
        help="Density-bin width; 0.05 matches the existing streaming result.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume caches whose scientific configuration matches this run.",
    )
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def resolution_tag(resolution):
    text = f"{resolution:g}".replace(".", "p")
    return f"{text}deg"


def build_registry(data_root, scenario):
    registry = {}
    for variable_key, spec in fgen.VARIABLE_SPECS.items():
        directory = data_root / spec["directory"] / "scenarios" / scenario
        for model, files in fgen.group_files_by_model(str(directory)).items():
            registry.setdefault(model, {})[variable_key] = files
    return registry


def file_signature(path):
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_input_manifest(args, registry, area_index):
    return {
        "models": {
            model: {
                "variables": {
                    key: [file_signature(path) for path in registry[model][key]]
                    for key in VARIABLE_KEYS
                },
                "areacello": file_signature(area_index[model][0]),
            }
            for model in args.models
        },
        "legacy_fgen": file_signature(args.legacy_fgen),
    }


def build_config(args, input_manifest):
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "data_root": str(args.data_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "legacy_fgen": str(args.legacy_fgen.resolve()),
        "scenario": args.scenario,
        "models": list(args.models),
        "variables": list(VARIABLE_KEYS),
        "resolution_degrees": float(args.resolution),
        "last_n_years": int(args.last_n_years),
        "last_n_months": int(args.last_n_months),
        "source_months": int(args.last_n_months),
        "rho_min": float(args.rho_min),
        "rho_max": float(args.rho_max),
        "step_size": float(args.step_size),
        "temporal_aggregation": "per-model calendar-month climatology",
        "model_mean": "equal model weight over joint four-variable finite values",
        "area_method": "mean source area assigned from each model's native tos grid",
        "regrid_method": "source_area_weighted_center_bin_average",
        "input_manifest": input_manifest,
    }


def config_fingerprint(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def new_checkpoint(config, fingerprint):
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "config": config,
        "config_fingerprint": fingerprint,
        "completed_variables": {model: [] for model in config["models"]},
        "completed_areas": [],
        "completed_models": [],
        "area_qc": {},
        "regrid_qc": {model: {} for model in config["models"]},
        "time_windows": {},
        "timings": [],
        "errors": [],
    }


def atomic_pickle_dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".part")
    with temporary_path.open("wb") as handle:
        pickle.dump(payload, handle)
    os.replace(temporary_path, path)


def save_checkpoint(path, checkpoint):
    checkpoint["updated_utc"] = utc_now()
    atomic_pickle_dump(path, checkpoint)


def load_or_create_checkpoint(args, config, fingerprint):
    checkpoint_path = args.output_dir / "checkpoint.pkl"
    if checkpoint_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"Checkpoint already exists at {checkpoint_path}; use --resume or a new output directory"
            )
        with checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if checkpoint.get("config_fingerprint") != fingerprint:
            raise ValueError(
                "Existing checkpoint configuration does not match this run. "
                "Use the original arguments or a different output directory."
            )
        return checkpoint_path, checkpoint

    generated_paths = [
        args.output_dir / "cache",
        *final_output_paths(args).values(),
        result_output_path(args),
    ]
    if not args.resume and any(path.exists() for path in generated_paths):
        raise FileExistsError(
            f"Generated output already exists under {args.output_dir}; use --resume or a new output directory"
        )

    checkpoint = new_checkpoint(config, fingerprint)
    save_checkpoint(checkpoint_path, checkpoint)
    return checkpoint_path, checkpoint


def validate_args(args):
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")
    if args.last_n_years <= 0 or args.last_n_months <= 0:
        raise ValueError("--last-n-years and --last-n-months must be positive")
    if args.last_n_months % 12 != 0:
        raise ValueError("--last-n-months must contain a whole number of years")
    if args.last_n_years * 12 < args.last_n_months:
        raise ValueError("--last-n-years is too short to provide --last-n-months")
    if args.time_chunk <= 0:
        raise ValueError("--time-chunk must be positive")
    if args.rho_max <= args.rho_min or args.step_size <= 0:
        raise ValueError("Invalid density range or --step-size")
    if not args.models or len(args.models) != len(set(args.models)):
        raise ValueError("--models must contain at least one unique model name")
    grids.standard_grid(args.resolution)


def validate_inputs(args, registry, area_index):
    errors = []
    for model in args.models:
        missing_variables = [key for key in VARIABLE_KEYS if key not in registry.get(model, {})]
        if missing_variables:
            errors.append(f"{model}: missing {', '.join(missing_variables)}")
        if model not in area_index:
            errors.append(f"{model}: no matching native areacello file")
    if errors:
        raise FileNotFoundError("Input validation failed:\n  " + "\n  ".join(errors))
    if not args.legacy_fgen.exists():
        raise FileNotFoundError(f"Legacy Fgen result not found: {args.legacy_fgen}")


def cache_directory(args, model):
    return args.output_dir / "cache" / model


def variable_cache_path(args, model, variable_key):
    return cache_directory(args, model) / f"{variable_key}_monthly_climatology.nc"


def area_cache_path(args, model):
    return cache_directory(args, model) / "areacello_binned.nc"


def final_output_paths(args):
    tag = resolution_tag(args.resolution)
    return {
        key: args.output_dir / f"MMM_{key}_binned_{tag}.nc"
        for key in VARIABLE_KEYS
    }


def result_output_path(args):
    return args.output_dir / f"MMM_Fgen_binned_{resolution_tag(args.resolution)}.pkl"


def data_variable_encoding(data_array):
    chunksizes = tuple(
        1 if dim == "time" else min(size, 90 if dim == "lat" else 180)
        for dim, size in zip(data_array.dims, data_array.shape)
    )
    encoding = {
        "zlib": True,
        "complevel": 4,
        "shuffle": True,
        "chunksizes": chunksizes,
    }
    if np.issubdtype(data_array.dtype, np.integer):
        encoding["_FillValue"] = None
    return encoding


def atomic_write_netcdf(dataset, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".part")
    temporary_path.unlink(missing_ok=True)
    encoding = {
        name: data_variable_encoding(data_array)
        for name, data_array in dataset.data_vars.items()
        if data_array.ndim > 0
    }
    dataset.to_netcdf(
        temporary_path,
        engine="h5netcdf",
        encoding=encoding,
    )
    os.replace(temporary_path, path)


def time_component(time_coordinate, component):
    try:
        values = np.asarray(getattr(time_coordinate.dt, component).values, dtype=np.int64)
    except Exception:
        values = np.asarray(
            [getattr(value, component) for value in time_coordinate.values],
            dtype=np.int64,
        )
    return values


def year_month_ids(time_coordinate):
    years = time_component(time_coordinate, "year")
    months = time_component(time_coordinate, "month")
    if years.ndim != 1 or months.ndim != 1 or years.size != months.size:
        raise ValueError("Time coordinate must be one-dimensional monthly data")
    if np.any((months < 1) | (months > 12)):
        raise ValueError("Time coordinate contains an invalid calendar month")
    ids = years * 12 + months - 1
    if np.unique(ids).size != ids.size:
        raise ValueError("Time coordinate contains duplicate year-month values")
    if ids.size > 1 and np.any(np.diff(ids) <= 0):
        raise ValueError("Time coordinate is not strictly increasing by year-month")
    return ids


def format_year_month(year_month_id):
    year, zero_based_month = divmod(int(year_month_id), 12)
    return f"{year:04d}-{zero_based_month + 1:02d}"


def select_common_trailing_months(
    model,
    opened_ds_map,
    backend_map,
    last_n_months,
):
    ids_by_variable = {}
    for variable_key in VARIABLE_KEYS:
        try:
            ids_by_variable[variable_key] = year_month_ids(
                opened_ds_map[variable_key]["time"]
            )
        except Exception as exc:
            raise ValueError(f"{model} {variable_key}: invalid time coordinate") from exc

    common_ids = set(ids_by_variable[VARIABLE_KEYS[0]].tolist())
    for variable_key in VARIABLE_KEYS[1:]:
        common_ids.intersection_update(ids_by_variable[variable_key].tolist())
    common_ids = np.asarray(sorted(common_ids), dtype=np.int64)
    if common_ids.size < last_n_months:
        raise ValueError(
            f"{model}: only {common_ids.size} year-months are shared by all variables; "
            f"need {last_n_months}"
        )

    split_points = np.flatnonzero(np.diff(common_ids) != 1) + 1
    common_runs = np.split(common_ids, split_points)
    latest_run = common_runs[-1]
    if latest_run.size < last_n_months:
        raise ValueError(
            f"{model}: latest common consecutive run has {latest_run.size} months; "
            f"need {last_n_months}"
        )
    selected_ids = latest_run[-last_n_months:]

    selected = {}
    for variable_key in VARIABLE_KEYS:
        ids = ids_by_variable[variable_key]
        first_matches = np.flatnonzero(ids == selected_ids[0])
        last_matches = np.flatnonzero(ids == selected_ids[-1])
        if first_matches.size != 1 or last_matches.size != 1:
            raise ValueError(f"{model} {variable_key}: unable to locate common time window")
        start = int(first_matches[0])
        stop = int(last_matches[0]) + 1
        if not np.array_equal(ids[start:stop], selected_ids):
            raise ValueError(
                f"{model} {variable_key}: common year-month window is not contiguous in the source"
            )
        dataset = opened_ds_map[variable_key].isel(time=slice(start, stop))
        if backend_map.get(variable_key) == "scipy":
            dataset = dataset.load()
        selected[variable_key] = dataset

    print(
        f"Aligning {model} to common year-month window "
        f"{format_year_month(selected_ids[0])} -> {format_year_month(selected_ids[-1])} "
        f"({last_n_months} months).",
        flush=True,
    )
    return selected, selected_ids


def validate_model_time(model, trimmed_ds_map, selected_ids, last_n_months):
    for variable_key in VARIABLE_KEYS:
        time_coordinate = trimmed_ds_map[variable_key]["time"]
        if time_coordinate.size != last_n_months:
            raise ValueError(
                f"{model}: expected {last_n_months} {variable_key} months, "
                f"found {time_coordinate.size}"
            )
        if not np.array_equal(year_month_ids(time_coordinate), selected_ids):
            raise ValueError(
                f"{model}: {variable_key} year-month coordinate does not match the common window"
            )

    months = (selected_ids % 12 + 1).astype(np.int16)
    expected_next = months[:-1] % 12 + 1
    if not np.array_equal(months[1:], expected_next):
        raise ValueError(f"{model}: selected time coordinate is not consecutive monthly data")

    expected_per_month = last_n_months // 12
    counts = np.bincount(months, minlength=13)[1:13]
    if not np.array_equal(counts, np.full(12, expected_per_month)):
        raise ValueError(
            f"{model}: expected {expected_per_month} samples per calendar month, "
            f"found {counts.tolist()}"
        )
    return months


def load_native_area(model, area_index):
    dataset = None
    try:
        dataset, _ = fgen.open_dataset_with_fallback(
            area_index[model][0],
            label=f"{model} areacello",
        )
        area = dataset["areacello"]
        if "time" in area.dims:
            area = area.isel(time=0, drop=True)
        return area.squeeze(drop=True).load()
    finally:
        fgen.safe_close(dataset)


def source_dataset_for_binning(model, variable_key, source_dataset, area_dataset):
    """Apply the documented GISS polar-coordinate correction for native wfo."""
    if variable_key != "wfo" or model not in GISS_WFO_AREA_COORDINATE_MODELS:
        return source_dataset, "none"

    source = source_dataset[variable_key]
    area = area_dataset["areacello"]
    if source.dims[-2:] != ("lat", "lon") or area.dims != ("lat", "lon"):
        raise ValueError(f"{model} wfo: expected rectilinear lat/lon source and area grids")
    if source.shape[-2:] != area.shape:
        raise ValueError(
            f"{model} wfo: source shape {source.shape[-2:]} does not match area {area.shape}"
        )

    source_lat = np.asarray(source_dataset["lat"].values, dtype=np.float64)
    source_lon = grids.normalize_longitude(
        np.asarray(source_dataset["lon"].values, dtype=np.float64)
    )
    area_lat = np.asarray(area_dataset["lat"].values, dtype=np.float64)
    area_lon = grids.normalize_longitude(
        np.asarray(area_dataset["lon"].values, dtype=np.float64)
    )
    tolerance = grids.COORDINATE_MATCH_TOLERANCE_DEGREES
    if not np.allclose(source_lon, area_lon, rtol=0.0, atol=tolerance):
        raise ValueError(f"{model} wfo: longitude does not match areacello")
    if not np.allclose(source_lat[1:-1], area_lat[1:-1], rtol=0.0, atol=tolerance):
        raise ValueError(f"{model} wfo: interior latitude does not match areacello")
    endpoint_difference = np.abs(source_lat[[0, -1]] - area_lat[[0, -1]])
    if np.any(endpoint_difference > 0.5 + tolerance):
        raise ValueError(
            f"{model} wfo: polar latitude difference exceeds the expected 0.5 degrees"
        )

    adjusted = source_dataset.assign_coords(
        lat=area_dataset["lat"],
        lon=area_dataset["lon"],
    )
    return adjusted, "GISS wfo polar coordinates replaced by matching areacello centers"


def climatology_dataset(
    model,
    variable_key,
    source_dataset,
    area_dataset,
    target,
    month_numbers,
    time_chunk,
    config,
    fingerprint,
    coordinate_adjustment="none",
):
    n_lat = target.sizes["lat"]
    n_lon = target.sizes["lon"]
    sums = np.zeros((12, n_lat, n_lon), dtype=np.float64)
    counts = np.zeros((12, n_lat, n_lon), dtype=np.uint16)
    maximum_area_relative_error = 0.0

    for start in range(0, month_numbers.size, time_chunk):
        stop = min(start + time_chunk, month_numbers.size)
        source_chunk = source_dataset.isel(time=slice(start, stop))
        regridded_result = grids.regrid_area_weighted_bins(
            source_chunk,
            variable_key,
            target,
            area_dataset,
        )
        assigned_area = np.asarray(
            regridded_result["source_area_sum"].sum(("lat", "lon")).values,
            dtype=np.float64,
        ).ravel()
        source_area = np.asarray(
            grids.source_valid_area_totals(
                source_chunk,
                variable_key,
                area_dataset,
            ),
            dtype=np.float64,
        ).ravel()
        valid_totals = np.isfinite(source_area) & (source_area > 0)
        relative_error = np.zeros(source_area.shape, dtype=np.float64)
        relative_error[valid_totals] = (
            np.abs(assigned_area[valid_totals] - source_area[valid_totals])
            / source_area[valid_totals]
        )
        chunk_maximum_error = float(np.nanmax(relative_error))
        maximum_area_relative_error = max(
            maximum_area_relative_error,
            chunk_maximum_error,
        )
        if chunk_maximum_error > AREA_RELATIVE_ERROR_TOLERANCE:
            raise ValueError(
                f"{model} {variable_key}: binned area relative error "
                f"{chunk_maximum_error:.6g} exceeds "
                f"{AREA_RELATIVE_ERROR_TOLERANCE:.6g} for months {start}:{stop}"
            )

        regridded = regridded_result[variable_key].transpose("time", "lat", "lon")
        values = np.asarray(regridded.values, dtype=np.float64)
        for local_index, month in enumerate(month_numbers[start:stop]):
            month_index = int(month) - 1
            finite = np.isfinite(values[local_index])
            sums[month_index][finite] += values[local_index][finite]
            counts[month_index][finite] += 1

    climatology = np.full(sums.shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=climatology, where=counts > 0)
    source_attrs = source_dataset[variable_key].attrs.copy()
    source_attrs.update(
        long_name=f"Per-model monthly climatology of binned {variable_key}",
        model=model,
        regrid_method="source_area_weighted_center_bin_average",
    )

    result = xr.Dataset(
        {
            variable_key: xr.DataArray(
                climatology,
                dims=("time", "lat", "lon"),
                coords={"time": MONTH_NUMBERS, "lat": target["lat"], "lon": target["lon"]},
                attrs=source_attrs,
            ),
            "sample_count": xr.DataArray(
                counts,
                dims=("time", "lat", "lon"),
                coords={"time": MONTH_NUMBERS, "lat": target["lat"], "lon": target["lon"]},
                attrs={
                    "long_name": "number of source years contributing to the model climatology",
                    "units": "1",
                },
            ),
        },
        attrs={
            "workflow": "raw field binned before per-model monthly climatology",
            "model": model,
            "variable": variable_key,
            "scenario": config["scenario"],
            "source_months": config["last_n_months"],
            "resolution_degrees": config["resolution_degrees"],
            "maximum_assigned_area_relative_error": maximum_area_relative_error,
            "source_period_start": format_year_month(
                year_month_ids(source_dataset["time"])[0]
            ),
            "source_period_end": format_year_month(
                year_month_ids(source_dataset["time"])[-1]
            ),
            "coordinate_adjustment": coordinate_adjustment,
            "config_fingerprint": fingerprint,
        },
    )
    result["time"].attrs.update(
        long_name="climatological calendar month",
        units="1",
        valid_min=1,
        valid_max=12,
    )
    return grids.as_standard_dataset(result, target), {
        "maximum_assigned_area_relative_error": maximum_area_relative_error,
        "finite_climatology_cells": int(np.count_nonzero(np.isfinite(climatology))),
        "minimum_sample_count": int(counts[counts > 0].min()) if np.any(counts > 0) else 0,
        "maximum_sample_count": int(counts.max()),
        "coordinate_adjustment": coordinate_adjustment,
    }


def binned_area_dataset(model, tos_dataset, area_dataset, target, config, fingerprint):
    template = xr.ones_like(
        tos_dataset["tos"].isel(time=0, drop=True),
        dtype=np.float64,
    ).rename("_ocean_template")
    template.attrs = {}
    binned = grids.regrid_area_weighted_bins(
        template.to_dataset(name=template.name),
        template.name,
        target,
        area_dataset,
    )
    assigned_area = binned["source_area_sum"].rename("areacello")
    source_area = grids.source_valid_area_totals(
        template.to_dataset(name=template.name),
        template.name,
        area_dataset,
    )[0]
    assigned_total = float(assigned_area.sum().values)
    relative_error = abs(assigned_total - source_area) / source_area
    if not np.isfinite(relative_error) or relative_error > AREA_RELATIVE_ERROR_TOLERANCE:
        raise ValueError(
            f"{model}: binned area relative error {relative_error:.6g} exceeds "
            f"{AREA_RELATIVE_ERROR_TOLERANCE:.6g}"
        )

    assigned_area.attrs.update(
        units=area_dataset["areacello"].attrs.get("units", "m2"),
        long_name=(
            "native ocean-cell area assigned to "
            f"{config['resolution_degrees']:g}-degree target cells"
        ),
        area_method="source cell-center bin sum from native tos grid",
    )
    result = assigned_area.to_dataset(name="areacello")
    result.attrs.update(
        workflow="native ocean area assigned to the common target grid",
        model=model,
        scenario=config["scenario"],
        resolution_degrees=config["resolution_degrees"],
        source_area_total_m2=float(source_area),
        assigned_area_total_m2=assigned_total,
        assigned_area_relative_error=float(relative_error),
        config_fingerprint=fingerprint,
    )
    return grids.as_standard_dataset(result, target), {
        "source_area_total_m2": float(source_area),
        "assigned_area_total_m2": assigned_total,
        "assigned_area_relative_error": float(relative_error),
    }


def valid_variable_cache(path, variable_key, target, fingerprint):
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path, engine="h5netcdf") as dataset:
            values = np.asarray(dataset[variable_key].values) if variable_key in dataset else None
            sample_count = (
                np.asarray(dataset["sample_count"].values)
                if "sample_count" in dataset
                else None
            )
            source_months = int(dataset.attrs.get("source_months", 0))
            maximum_samples = source_months // 12
            return bool(
                variable_key in dataset
                and "sample_count" in dataset
                and dataset[variable_key].dims == ("time", "lat", "lon")
                and dataset["sample_count"].dims == ("time", "lat", "lon")
                and dataset.sizes.get("time") == 12
                and dataset.sizes.get("lat") == target.sizes["lat"]
                and dataset.sizes.get("lon") == target.sizes["lon"]
                and dataset.sizes.get("lat_b") == target.sizes["lat_b"]
                and dataset.sizes.get("lon_b") == target.sizes["lon_b"]
                and np.array_equal(dataset["time"].values, MONTH_NUMBERS)
                and np.array_equal(dataset["lat"].values, target["lat"].values)
                and np.array_equal(dataset["lon"].values, target["lon"].values)
                and np.issubdtype(dataset["sample_count"].dtype, np.integer)
                and source_months > 0
                and source_months % 12 == 0
                and np.all(np.isfinite(values) | np.isnan(values))
                and np.all(sample_count >= 0)
                and np.all(sample_count <= maximum_samples)
                and np.array_equal(np.isfinite(values), sample_count > 0)
                and dataset.attrs.get("config_fingerprint") == fingerprint
            )
    except Exception:
        return False


def valid_area_cache(path, target, fingerprint):
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path, engine="h5netcdf") as dataset:
            area = (
                np.asarray(dataset["areacello"].values, dtype=np.float64)
                if "areacello" in dataset
                else None
            )
            return bool(
                "areacello" in dataset
                and dataset["areacello"].dims == ("lat", "lon")
                and dataset.sizes.get("lat") == target.sizes["lat"]
                and dataset.sizes.get("lon") == target.sizes["lon"]
                and dataset.sizes.get("lat_b") == target.sizes["lat_b"]
                and dataset.sizes.get("lon_b") == target.sizes["lon_b"]
                and np.array_equal(dataset["lat"].values, target["lat"].values)
                and np.array_equal(dataset["lon"].values, target["lon"].values)
                and np.all(np.isfinite(area))
                and np.all(area >= 0.0)
                and float(area.sum()) > 0.0
                and dataset.attrs.get("config_fingerprint") == fingerprint
            )
    except Exception:
        return False


def record_error(checkpoint, model, stage, exc):
    checkpoint["errors"].append(
        {
            "timestamp_utc": utc_now(),
            "model": model,
            "stage": stage,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    )


def process_model(
    args,
    model,
    registry,
    area_index,
    target,
    config,
    fingerprint,
    checkpoint_path,
    checkpoint,
):
    opened_ds_map = None
    trimmed_ds_map = None
    area_da = None
    started = time.perf_counter()
    try:
        print(f"\n=== Regridding {model} ===", flush=True)
        opened_ds_map, backend_map = fgen.open_model_inputs(
            model,
            registry,
            time_chunk=args.time_chunk,
        )
        trimmed_ds_map, selected_ids = select_common_trailing_months(
            model=model,
            opened_ds_map=opened_ds_map,
            backend_map=backend_map,
            last_n_months=args.last_n_months,
        )
        month_numbers = validate_model_time(
            model,
            trimmed_ds_map,
            selected_ids,
            args.last_n_months,
        )
        checkpoint["time_windows"][model] = {
            "start": format_year_month(selected_ids[0]),
            "end": format_year_month(selected_ids[-1]),
            "months": int(selected_ids.size),
        }
        save_checkpoint(checkpoint_path, checkpoint)
        area_da = load_native_area(model, area_index)
        area_ds = area_da.rename("areacello").to_dataset(name="areacello")

        completed_variables = set(checkpoint["completed_variables"].setdefault(model, []))
        for variable_key in VARIABLE_KEYS:
            path = variable_cache_path(args, model, variable_key)
            if variable_key in completed_variables and valid_variable_cache(
                path, variable_key, target, fingerprint
            ):
                print(f"  {variable_key}: valid cache found", flush=True)
                continue

            variable_started = time.perf_counter()
            source_for_binning, coordinate_adjustment = source_dataset_for_binning(
                model,
                variable_key,
                trimmed_ds_map[variable_key],
                area_ds,
            )
            dataset, regrid_qc = climatology_dataset(
                model=model,
                variable_key=variable_key,
                source_dataset=source_for_binning,
                area_dataset=area_ds,
                target=target,
                month_numbers=month_numbers,
                time_chunk=args.time_chunk,
                config=config,
                fingerprint=fingerprint,
                coordinate_adjustment=coordinate_adjustment,
            )
            atomic_write_netcdf(dataset, path)
            fgen.safe_close(dataset)
            completed_variables.add(variable_key)
            checkpoint["completed_variables"][model] = sorted(completed_variables)
            checkpoint["regrid_qc"].setdefault(model, {})[variable_key] = regrid_qc
            checkpoint["timings"].append(
                {
                    "model": model,
                    "stage": f"regrid_{variable_key}",
                    "seconds": time.perf_counter() - variable_started,
                }
            )
            save_checkpoint(checkpoint_path, checkpoint)
            print(f"  {variable_key}: wrote {path}", flush=True)

        area_path = area_cache_path(args, model)
        completed_areas = set(checkpoint["completed_areas"])
        if model not in completed_areas or not valid_area_cache(area_path, target, fingerprint):
            area_result, area_qc = binned_area_dataset(
                model=model,
                tos_dataset=trimmed_ds_map["tos"],
                area_dataset=area_ds,
                target=target,
                config=config,
                fingerprint=fingerprint,
            )
            atomic_write_netcdf(area_result, area_path)
            fgen.safe_close(area_result)
            completed_areas.add(model)
            checkpoint["completed_areas"] = sorted(completed_areas)
            checkpoint["area_qc"][model] = area_qc
            save_checkpoint(checkpoint_path, checkpoint)
            print(
                f"  area: wrote {area_path} "
                f"(relative error {area_qc['assigned_area_relative_error']:.3g})",
                flush=True,
            )

        if set(checkpoint["completed_variables"][model]) == set(VARIABLE_KEYS) and model in set(
            checkpoint["completed_areas"]
        ):
            completed_models = set(checkpoint["completed_models"])
            completed_models.add(model)
            checkpoint["completed_models"] = sorted(completed_models)
            checkpoint["timings"].append(
                {
                    "model": model,
                    "stage": "model_total",
                    "seconds": time.perf_counter() - started,
                }
            )
            save_checkpoint(checkpoint_path, checkpoint)
            print(f"Completed {model}", flush=True)

    except Exception as exc:
        record_error(checkpoint, model, "model_regrid", exc)
        save_checkpoint(checkpoint_path, checkpoint)
        print(f"ERROR {model}: {exc!r}", flush=True)
    finally:
        if opened_ds_map is not None:
            for dataset in opened_ds_map.values():
                fgen.safe_close(dataset)
        if trimmed_ds_map is not None:
            for dataset in trimmed_ds_map.values():
                fgen.safe_close(dataset)
        del opened_ds_map, trimmed_ds_map, area_da
        gc.collect()


def aggregate_model_climatologies(args, target, config, fingerprint):
    shape = (12, target.sizes["lat"], target.sizes["lon"])
    sums = {key: np.zeros(shape, dtype=np.float64) for key in VARIABLE_KEYS}
    model_count = np.zeros(shape, dtype=np.uint16)
    area_sum = np.zeros(shape[1:], dtype=np.float64)
    variable_units = {key: None for key in VARIABLE_KEYS}
    area_units = None
    coordinate_adjustments = {}

    for model in args.models:
        with ExitStack() as stack:
            datasets = {
                key: stack.enter_context(
                    xr.open_dataset(variable_cache_path(args, model, key), engine="h5netcdf")
                )
                for key in VARIABLE_KEYS
            }
            values = {
                key: np.asarray(datasets[key][key].values, dtype=np.float64)
                for key in VARIABLE_KEYS
            }
            for key in VARIABLE_KEYS:
                units = datasets[key][key].attrs.get("units")
                if not units:
                    raise ValueError(f"{model} {key}: source units are missing from the cache")
                if variable_units[key] is None:
                    variable_units[key] = units
                elif units != variable_units[key]:
                    raise ValueError(
                        f"{model} {key}: units {units!r} do not match "
                        f"{variable_units[key]!r}"
                    )
                adjustment = datasets[key].attrs.get("coordinate_adjustment", "none")
                if adjustment != "none":
                    coordinate_adjustments[f"{model}:{key}"] = adjustment
            joint_valid = np.logical_and.reduce(
                [np.isfinite(values[key]) for key in VARIABLE_KEYS]
            )
            model_count[joint_valid] += 1
            for key in VARIABLE_KEYS:
                sums[key][joint_valid] += values[key][joint_valid]

            area_dataset = stack.enter_context(
                xr.open_dataset(area_cache_path(args, model), engine="h5netcdf")
            )
            model_area = np.asarray(area_dataset["areacello"].values, dtype=np.float64)
            model_area_units = area_dataset["areacello"].attrs.get("units", "m2")
            if area_units is None:
                area_units = model_area_units
            elif model_area_units != area_units:
                raise ValueError(
                    f"{model}: areacello units {model_area_units!r} do not match {area_units!r}"
                )
            area_sum += np.nan_to_num(model_area, nan=0.0)

    mmm_values = {}
    for key in VARIABLE_KEYS:
        mean = np.full(shape, np.nan, dtype=np.float64)
        np.divide(sums[key], model_count, out=mean, where=model_count > 0)
        mmm_values[key] = mean
    mean_area = area_sum / len(args.models)

    common_attrs = {
        "workflow": "binned raw fields then per-model monthly climatology then equal-model mean",
        "scenario": config["scenario"],
        "models": json.dumps(list(args.models)),
        "model_count_total": len(args.models),
        "source_months_per_model": config["last_n_months"],
        "temporal_aggregation": config["temporal_aggregation"],
        "model_mean": config["model_mean"],
        "area_method": config["area_method"],
        "regrid_method": config["regrid_method"],
        "source_coordinate_adjustments": json.dumps(
            coordinate_adjustments,
            sort_keys=True,
        ),
        "resolution_degrees": config["resolution_degrees"],
        "config_fingerprint": fingerprint,
        "created_utc": utc_now(),
    }
    output_paths = final_output_paths(args)
    for key, path in output_paths.items():
        dataset = xr.Dataset(
            {
                key: xr.DataArray(
                    mmm_values[key],
                    dims=("time", "lat", "lon"),
                    coords={
                        "time": MONTH_NUMBERS,
                        "lat": target["lat"],
                        "lon": target["lon"],
                    },
                    attrs={
                        "long_name": f"Equal-model-mean monthly climatology of binned {key}",
                        "source_variable": key,
                        "units": variable_units[key],
                    },
                ),
                "model_count": xr.DataArray(
                    model_count,
                    dims=("time", "lat", "lon"),
                    coords={
                        "time": MONTH_NUMBERS,
                        "lat": target["lat"],
                        "lon": target["lon"],
                    },
                    attrs={
                        "long_name": "models jointly contributing all four MMM input variables",
                        "units": "1",
                    },
                ),
                "areacello": xr.DataArray(
                    mean_area,
                    dims=("lat", "lon"),
                    coords={"lat": target["lat"], "lon": target["lon"]},
                    attrs={
                        "long_name": "mean native ocean area assigned to target cells",
                        "units": area_units or "m2",
                        "area_method": config["area_method"],
                    },
                ),
            },
            attrs={**common_attrs, "variable": key},
        )
        dataset["time"].attrs.update(
            long_name="climatological calendar month",
            units="1",
            valid_min=1,
            valid_max=12,
        )
        dataset = grids.as_standard_dataset(dataset, target)
        atomic_write_netcdf(dataset, path)
        fgen.safe_close(dataset)
        print(f"Wrote final MMM field: {path}", flush=True)

    positive_counts = model_count[model_count > 0]
    qc = {
        "model_count_min": int(positive_counts.min()) if positive_counts.size else 0,
        "model_count_max": int(positive_counts.max()) if positive_counts.size else 0,
        "model_count_zero_cells": int(np.count_nonzero(model_count == 0)),
        "mean_area_total_m2": float(mean_area.sum()),
        "coordinate_adjustments": coordinate_adjustments,
    }
    return output_paths, qc


def validate_final_outputs(output_paths, target, expected_models):
    with ExitStack() as stack:
        datasets = {
            key: stack.enter_context(xr.open_dataset(path, engine="h5netcdf"))
            for key, path in output_paths.items()
        }
        reference = datasets["tos"]
        reference_count = np.asarray(reference["model_count"].values)
        reference_area = np.asarray(reference["areacello"].values)
        for key in VARIABLE_KEYS:
            dataset = datasets[key]
            if dataset[key].dims != ("time", "lat", "lon"):
                raise ValueError(f"Final {key} dimensions are {dataset[key].dims}")
            if dataset[key].shape != (
                12,
                target.sizes["lat"],
                target.sizes["lon"],
            ):
                raise ValueError(f"Final {key} shape is {dataset[key].shape}")
            if not np.array_equal(dataset["time"].values, MONTH_NUMBERS):
                raise ValueError(f"Final {key} calendar-month ordering is invalid")
            for coordinate in ("lat", "lon", "lat_b", "lon_b"):
                if not np.array_equal(dataset[coordinate].values, target[coordinate].values):
                    raise ValueError(f"Final {key} {coordinate} coordinate is invalid")
            if not np.array_equal(dataset["model_count"].values, reference_count):
                raise ValueError(f"Final {key} model_count differs from tos")
            if not np.array_equal(dataset["areacello"].values, reference_area):
                raise ValueError(f"Final {key} areacello differs from tos")
            if int(np.nanmax(dataset["model_count"].values)) > expected_models:
                raise ValueError(f"Final {key} model_count exceeds the requested cohort")
        if not np.issubdtype(reference["model_count"].dtype, np.integer):
            raise ValueError("Final model_count is not integer-valued")
        if np.any(~np.isfinite(reference_area)) or np.any(reference_area < 0):
            raise ValueError("Final areacello contains invalid values")


def calculate_mmm_fgen(args, output_paths):
    opened = {}
    fsurf = None
    try:
        for key, path in output_paths.items():
            opened[key] = xr.open_dataset(path, engine="h5netcdf")
        area = opened["tos"]["areacello"].load()
        fsurf = fgen.compute_fsurf("MMM", opened, area).load()
        rho_classes = np.arange(
            args.rho_min - args.step_size,
            args.rho_max + args.step_size,
            args.step_size,
        )
        result = fgen.compute_fgen_for_model(
            model="MMM",
            fsurf_ds=fsurf,
            area_da=area,
            rho_classes=rho_classes,
            step_size=args.step_size,
        )
        if result is None or result.empty:
            raise ValueError("MMM calculation produced no Fgen rows")
        result.attrs.update(
            model="MMM",
            method="binned_raw_fields_then_multi_model_mean",
            resolution_degrees=args.resolution,
        )
        return result
    finally:
        for dataset in opened.values():
            fgen.safe_close(dataset)
        fgen.safe_close(fsurf)


def validate_legacy_density_grid(legacy_path, models, result):
    with legacy_path.open("rb") as handle:
        legacy = pickle.load(handle)
    if not isinstance(legacy, dict):
        raise ValueError(f"Unexpected legacy Fgen payload in {legacy_path}")
    missing = [model for model in models if model not in legacy]
    if missing:
        raise KeyError(f"Legacy Fgen result is missing models: {missing}")

    result_rho = result["rho"].to_numpy()
    for model in models:
        profile = legacy[model]
        if "rho" not in profile or not np.array_equal(profile["rho"].to_numpy(), result_rho):
            raise ValueError(
                f"Legacy density bins for {model} do not exactly match the MMM result"
            )


def main():
    args = parse_args()
    validate_args(args)
    target = grids.standard_grid(args.resolution)
    registry = build_registry(args.data_root, args.scenario)
    area_index = fgen.group_files_by_model(str(args.data_root / "areacello"))
    validate_inputs(args, registry, area_index)
    input_manifest = build_input_manifest(args, registry, area_index)
    config = build_config(args, input_manifest)
    fingerprint = config_fingerprint(config)
    checkpoint_path, checkpoint = load_or_create_checkpoint(
        args, config, fingerprint
    )

    print(f"Models ({len(args.models)}): {args.models}", flush=True)
    print(
        f"Target grid: {target.sizes['lat']}x{target.sizes['lon']} at "
        f"{args.resolution:g} degree",
        flush=True,
    )
    print(f"Output directory: {args.output_dir}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)

    for model in args.models:
        if model in set(checkpoint["completed_models"]):
            caches_valid = all(
                valid_variable_cache(
                    variable_cache_path(args, model, key), key, target, fingerprint
                )
                for key in VARIABLE_KEYS
            ) and valid_area_cache(area_cache_path(args, model), target, fingerprint)
            if caches_valid:
                print(f"\n=== {model}: all caches complete ===", flush=True)
                continue
            checkpoint["completed_models"] = [
                completed
                for completed in checkpoint["completed_models"]
                if completed != model
            ]
            save_checkpoint(checkpoint_path, checkpoint)

        process_model(
            args=args,
            model=model,
            registry=registry,
            area_index=area_index,
            target=target,
            config=config,
            fingerprint=fingerprint,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
        )

    missing_models = sorted(set(args.models) - set(checkpoint["completed_models"]))
    if missing_models:
        raise RuntimeError(
            "Not all requested models completed; resume after addressing failures: "
            + ", ".join(missing_models)
        )

    aggregation_started = time.perf_counter()
    output_paths, aggregation_qc = aggregate_model_climatologies(
        args, target, config, fingerprint
    )
    validate_final_outputs(output_paths, target, len(args.models))
    result = calculate_mmm_fgen(args, output_paths)
    validate_legacy_density_grid(args.legacy_fgen, args.models, result)

    aggregation_timing = {
        "model": "MMM",
        "stage": "aggregate_and_fgen",
        "seconds": time.perf_counter() - aggregation_started,
    }
    checkpoint["timings"].append(aggregation_timing)

    result_payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "metadata": {
            **config,
            "created_utc": utc_now(),
            "models_processed": list(args.models),
            "output_files": {key: str(path) for key, path in output_paths.items()},
            "result_file": str(result_output_path(args)),
            "legacy_fgen_file": str(args.legacy_fgen),
            **aggregation_qc,
        },
        "result": result,
        "area_qc": checkpoint["area_qc"],
        "regrid_qc": checkpoint["regrid_qc"],
        "time_windows": checkpoint["time_windows"],
        "timings": checkpoint["timings"],
        "errors": checkpoint["errors"],
    }
    atomic_pickle_dump(result_output_path(args), result_payload)
    checkpoint["final_result"] = str(result_output_path(args))
    checkpoint["completed_utc"] = utc_now()
    save_checkpoint(checkpoint_path, checkpoint)

    print(f"Wrote MMM Fgen result: {result_output_path(args)}", flush=True)
    print(
        f"Contributing-model count range: {aggregation_qc['model_count_min']} -> "
        f"{aggregation_qc['model_count_max']}",
        flush=True,
    )


if __name__ == "__main__":
    with dask.config.set({"array.slicing.split_large_chunks": True}):
        main()
