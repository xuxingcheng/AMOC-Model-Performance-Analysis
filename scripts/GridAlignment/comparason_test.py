#!/usr/bin/env python
"""Compare native-grid and aligned-grid Fgen calculations on a small model batch.

The nonlinear diagnostics (rho and fsurf) are always calculated on each model's
native grid first. The comparison then either integrates them on the native grid
or regrids the diagnostics to the shared regular lat/lon grid before applying the
same Fgen density-bin integration.
"""

from __future__ import annotations

import argparse
import gc
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import dask
import numpy as np
import xarray as xr


SCRIPT_DIR = Path(__file__).resolve().parent
FGEN_DIR = SCRIPT_DIR.parent / "FgenCalculation"
if str(FGEN_DIR) not in sys.path:
    sys.path.insert(0, str(FGEN_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import Fgenrun2_streaming as fgen
import grids


DEFAULT_MODELS = ("E3SM-1-0", "MIROC6", "ICON-ESM-LR")
DEFAULT_METHODS = ("original", "nearest", "binned")
DEFAULT_OUTPUT = Path("/glade/derecho/scratch/stevenxu/tmp/gridtest_Fgen.pkl")
FSURF_VARIABLES = ("fsurf", "rho", "heat_comp", "fw_comp")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=fgen.DEFAULT_SCENARIO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--last-n-years", type=int, default=2)
    parser.add_argument("--last-n-months", type=int, default=12)
    parser.add_argument("--time-chunk", type=int, default=None)
    parser.add_argument("--rho-min", type=float, default=fgen.DEFAULT_RHO_MIN)
    parser.add_argument("--rho-max", type=float, default=fgen.DEFAULT_RHO_MAX)
    parser.add_argument("--step-size", type=float, default=fgen.DEFAULT_STEP_SIZE)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_payload(args):
    return {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scenario": args.scenario,
            "models_requested": list(args.models),
            "methods_requested": list(args.methods),
            "resolution_degrees": args.resolution,
            "last_n_years": args.last_n_years,
            "last_n_months": args.last_n_months,
            "rho_min": args.rho_min,
            "rho_max": args.rho_max,
            "step_size": args.step_size,
            "workflow": (
                "Compute rho/fsurf on the native grid, then compare native-grid "
                "integration with nearest and area-binned regridding of diagnostics."
            ),
        },
        "results": {method: {} for method in args.methods},
        "timings": [],
        "errors": [],
    }


def load_or_create_payload(args):
    if not args.resume or not args.output.exists():
        return build_payload(args)

    with args.output.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError(f"Unexpected comparison output format in {args.output}")
    for method in args.methods:
        payload["results"].setdefault(method, {})
    payload.setdefault("timings", [])
    payload.setdefault("errors", [])
    return payload


def save_payload(output_path, payload):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(payload, handle)
    temporary_path.replace(output_path)


def area_dataset(area_da):
    area = area_da.rename("areacello")
    return area.to_dataset(name="areacello")


def standard_area_from_native(fsurf_ds, native_area_ds, target_grid):
    template = xr.ones_like(
        fsurf_ds["rho"].isel(time=0, drop=True),
        dtype=np.float64,
    ).rename("_ocean_template")
    template.attrs = {}
    binned = grids.regrid_area_weighted_bins(
        template.to_dataset(name=template.name),
        template.name,
        target_grid,
        native_area_ds,
    )
    standard_area = binned["source_area_sum"].rename("areacello")
    standard_area.attrs.update(
        units=native_area_ds["areacello"].attrs.get("units", "m2"),
        long_name="native ocean-cell area assigned to standard lat/lon cells",
        area_method="source cell-center bin sum",
    )
    return standard_area


def align_fsurf_nearest(fsurf_ds, native_area_ds, target_grid):
    aligned = {
        variable: grids.regrid_nearest(
            fsurf_ds,
            variable,
            target_grid,
            area_dataset=native_area_ds,
        )
        for variable in FSURF_VARIABLES
    }
    return xr.Dataset(aligned, attrs={"grid_alignment_method": "nearest"})


def align_fsurf_binned(fsurf_ds, native_area_ds, target_grid):
    aligned = {}
    for variable in FSURF_VARIABLES:
        result = grids.regrid_area_weighted_bins(
            fsurf_ds,
            variable,
            target_grid,
            native_area_ds,
        )
        aligned[variable] = result[variable]
    return xr.Dataset(aligned, attrs={"grid_alignment_method": "binned"})


def calculate_method_fgen(
    method,
    model,
    native_fsurf,
    native_area,
    native_area_ds,
    target_grid,
    standard_area,
    rho_classes,
    step_size,
):
    aligned_fsurf = None
    try:
        if method == "original":
            method_fsurf = native_fsurf
            method_area = native_area
        elif method == "nearest":
            aligned_fsurf = align_fsurf_nearest(native_fsurf, native_area_ds, target_grid)
            method_fsurf = aligned_fsurf
            method_area = standard_area
        elif method == "binned":
            aligned_fsurf = align_fsurf_binned(native_fsurf, native_area_ds, target_grid)
            method_fsurf = aligned_fsurf
            method_area = standard_area
        else:
            raise ValueError(f"Unsupported method: {method}")

        result = fgen.compute_fgen_for_model(
            model=model,
            fsurf_ds=method_fsurf,
            area_da=method_area,
            rho_classes=rho_classes,
            step_size=step_size,
        )
        if result is None or result.empty:
            raise ValueError(f"{model} {method}: no Fgen rows produced")
        result.attrs.update(
            model=model,
            method=method,
            resolution_degrees=None if method == "original" else target_grid.attrs["resolution_degrees"],
        )
        return result
    finally:
        fgen.safe_close(aligned_fsurf)


def record_error(payload, model, method, stage, exc):
    error = {
        "model": model,
        "method": method,
        "stage": stage,
        "error": repr(exc),
    }
    payload["errors"].append(error)
    print(f"ERROR {model} {method} {stage}: {exc!r}")


def get_time_chunk(args):
    if args.time_chunk is not None:
        return args.time_chunk
    return max(12, args.last_n_months)


def main():
    args = parse_args()
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")
    if args.last_n_months <= 0:
        raise ValueError("--last-n-months must be positive")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")

    payload = load_or_create_payload(args)
    registry = fgen.build_model_file_registry(args.scenario)
    area_index = fgen.group_files_by_model(fgen.AREA_DIR)
    models = fgen.get_candidate_models(
        registry=registry,
        include_models=args.models,
        exclude_models=[],
    )
    if args.max_models is not None:
        models = models[: args.max_models]

    target_grid = grids.standard_grid(args.resolution)
    rho_classes = np.arange(
        args.rho_min - args.step_size,
        args.rho_max + args.step_size,
        args.step_size,
    )
    time_chunk = get_time_chunk(args)

    print(f"Models: {models}")
    print(f"Methods: {args.methods}")
    print(f"Months per model: {args.last_n_months}")
    print(f"Standard grid: {target_grid.sizes['lat']}x{target_grid.sizes['lon']}")
    print(f"Output: {args.output}")

    for model in models:
        pending_methods = [
            method for method in args.methods if model not in payload["results"][method]
        ]
        if not pending_methods:
            print(f"\n=== {model}: all requested methods already complete ===")
            continue

        print(f"\n=== Processing {model}: {pending_methods} ===")
        opened_ds_map = None
        trimmed_ds_map = None
        native_area = None
        native_fsurf = None

        try:
            opened_ds_map, backend_map = fgen.open_model_inputs(
                model,
                registry,
                time_chunk=time_chunk,
            )
            trimmed_ds_map = fgen.align_and_trim_inputs(
                model=model,
                opened_ds_map=opened_ds_map,
                backend_map=backend_map,
                last_n_years=args.last_n_years,
                last_n_months=args.last_n_months,
            )
            native_area = fgen.load_area_for_model(model, area_index)
            if native_area is None:
                raise FileNotFoundError(f"{model}: no areacello file or alias")

            native_fsurf = fgen.compute_fsurf(model, trimmed_ds_map, native_area).load()
            native_area_ds = area_dataset(native_area)
            standard_area = None

            for method in pending_methods:
                started = time.perf_counter()
                try:
                    if method != "original" and standard_area is None:
                        standard_area = standard_area_from_native(
                            native_fsurf,
                            native_area_ds,
                            target_grid,
                        )
                    result = calculate_method_fgen(
                        method=method,
                        model=model,
                        native_fsurf=native_fsurf,
                        native_area=native_area,
                        native_area_ds=native_area_ds,
                        target_grid=target_grid,
                        standard_area=standard_area,
                        rho_classes=rho_classes,
                        step_size=args.step_size,
                    )
                    payload["results"][method][model] = result
                    elapsed = time.perf_counter() - started
                    payload["timings"].append(
                        {"model": model, "method": method, "seconds": elapsed}
                    )
                    save_payload(args.output, payload)
                    print(
                        f"Completed {model} {method}: {len(result)} rho bins, "
                        f"min Fgen={result['Fgen'].min():.6g}, {elapsed:.2f} s"
                    )
                except Exception as exc:
                    record_error(payload, model, method, "method", exc)
                    save_payload(args.output, payload)

        except Exception as exc:
            record_error(payload, model, "all", "model_setup", exc)
            save_payload(args.output, payload)
        finally:
            if opened_ds_map is not None:
                for dataset in opened_ds_map.values():
                    fgen.safe_close(dataset)
            if trimmed_ds_map is not None:
                for dataset in trimmed_ds_map.values():
                    fgen.safe_close(dataset)
            fgen.safe_close(native_fsurf)
            del opened_ds_map, trimmed_ds_map, native_area, native_fsurf
            gc.collect()

    payload["metadata"]["completed_utc"] = datetime.now(timezone.utc).isoformat()
    payload["metadata"]["models_processed"] = {
        method: sorted(payload["results"][method]) for method in args.methods
    }
    save_payload(args.output, payload)
    print("\nCompleted comparison.")
    for method in args.methods:
        print(f"  {method}: {len(payload['results'][method])} models")
    print(f"  errors: {len(payload['errors'])}")


if __name__ == "__main__":
    with dask.config.set({"array.slicing.split_large_chunks": True}):
        main()
