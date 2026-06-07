#!/usr/bin/env python
"""Compare regular-grid methods on a small batch of representative CMIP models."""

import argparse
import glob
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from grids import (
    describe_grid,
    open_dataset,
    regrid_area_weighted_bins,
    regrid_nearest,
    regrid_with_xesmf,
    select_first_steps,
    source_valid_area_totals,
    standard_grid,
    summarize,
)


DATA_ROOT = Path("/glade/work/stevenxu/AMOC_models")
DEFAULT_MODELS = ("E3SM-1-0", "MIROC6", "ICON-ESM-LR")
VARIABLE_DIRECTORIES = {
    "tos": "sea_surface_temperature",
    "sos": "sea_surface_salinity",
    "hfds": "heatflux",
    "wfo": "waterflux",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--variable", default="tos", choices=sorted(VARIABLE_DIRECTORIES))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "nearest",
            "binned",
            "xesmf-bilinear",
            "xesmf-nearest",
            "xesmf-conservative",
        ),
        default=("nearest", "binned"),
    )
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--time-steps", type=int, default=1)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def first_model_file(model: str, variable: str) -> str:
    directory = DATA_ROOT / VARIABLE_DIRECTORIES[variable] / "scenarios" / "PIControl"
    matches = sorted(glob.glob(str(directory / f"{variable}_*_{model}_*.nc")))
    if not matches:
        raise FileNotFoundError(f"No {variable} file found for {model}")
    return matches[0]


def area_file(model: str) -> str | None:
    matches = sorted(glob.glob(str(DATA_ROOT / "areacello" / f"areacello_*_{model}_*.nc")))
    return matches[0] if matches else None


def run_method(method, source, variable, target, model):
    area_path = area_file(model)
    area_context = open_dataset(area_path) if area_path else nullcontext(None)
    with area_context as area:
        if method == "nearest":
            return (
                regrid_nearest(source, variable, target, area_dataset=area),
                None,
                None,
            )
        if method == "binned":
            if area is None:
                raise FileNotFoundError(f"No areacello file found for {model}")
            result = regrid_area_weighted_bins(source, variable, target, area)
            source_area = source_valid_area_totals(source, variable, area)
            return result[variable], result["source_area_sum"], source_area
        if method == "xesmf-bilinear":
            return (
                regrid_with_xesmf(
                    source,
                    variable,
                    target,
                    method="bilinear",
                    area_dataset=area,
                ),
                None,
                None,
            )
        if method == "xesmf-nearest":
            return (
                regrid_with_xesmf(
                    source,
                    variable,
                    target,
                    method="nearest_s2d",
                    area_dataset=area,
                ),
                None,
                None,
            )
        if method == "xesmf-conservative":
            return (
                regrid_with_xesmf(
                    source,
                    variable,
                    target,
                    method="conservative_normed",
                    area_dataset=area,
                ),
                None,
                None,
            )
    raise ValueError(method)


def main():
    args = parse_args()
    target = standard_grid(args.resolution)
    report = []

    for model in args.models:
        source_path = first_model_file(model, args.variable)
        with open_dataset(source_path) as source:
            source = select_first_steps(source, args.variable, args.time_steps)
            grid = describe_grid(source, args.variable)
            for method in args.methods:
                row = {"model": model, "method": method, "source_grid": grid}
                started = time.perf_counter()
                try:
                    result, assigned_area, source_area = run_method(
                        method,
                        source,
                        args.variable,
                        target,
                        model,
                    )
                    row.update(summarize(result))
                    row["seconds"] = time.perf_counter() - started
                    row["standard_dims"] = result.dims[-2:] == ("lat", "lon")
                    if assigned_area is not None:
                        area_totals = assigned_area.sum(("lat", "lon")).values
                        row["assigned_area_min"] = float(np.nanmin(area_totals))
                        row["assigned_area_max"] = float(np.nanmax(area_totals))
                        relative_error = np.abs(area_totals.ravel() - source_area) / source_area
                        row["area_conservation_relative_error_max"] = float(
                            np.nanmax(relative_error)
                        )
                    print(
                        f"{model:16} {grid['grid_type']:13} {method:15} "
                        f"shape={result.shape} finite={row['finite_fraction']:.3f} "
                        f"seconds={row['seconds']:.2f}"
                    )
                except Exception as exc:
                    row["error"] = repr(exc)
                    row["seconds"] = time.perf_counter() - started
                    print(f"{model:16} {grid['grid_type']:13} {method:15} ERROR: {exc!r}")
                report.append(row)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, default=list) + "\n")
        print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
