#!/usr/bin/env python3
"""Actually regrid one timestep per model and print each resulting grid size."""

import argparse
from contextlib import ExitStack
from pathlib import Path

from grids import (
    open_dataset,
    regrid_area_weighted_bins,
    regrid_nearest,
    regridder,
    select_first_steps,
    standard_grid,
)


DATA_ROOT = Path("/glade/work/stevenxu/AMOC_models")
VARIABLE_DIRECTORIES = {
    "tos": "sea_surface_temperature",
    "sos": "sea_surface_salinity",
    "hfds": "heatflux",
    "wfo": "waterflux",
}
AREA_ALIASES = {
    "FGOALS-f3-L": "ACCESS-CM2",
    "FGOALS-g3": "ACCESS-CM2",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--scenario", default="PIControl")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--method",
        choices=("nearest", "binned", "regridder"),
        default="binned",
        help="Regridding method to run (default: binned).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Target resolution in degrees (default: 1).",
    )
    return parser.parse_args()


def first_files_by_model(directory: Path, variable: str):
    files = {}
    for path in sorted(directory.glob(f"{variable}_*.nc")):
        parts = path.name.split("_")
        if len(parts) >= 3:
            files.setdefault(parts[2], path)
    return files


def build_model_files(data_root: Path, scenario: str):
    return {
        variable: first_files_by_model(
            data_root / directory / "scenarios" / scenario,
            variable,
        )
        for variable, directory in VARIABLE_DIRECTORIES.items()
    }


def find_area_file(model: str, area_files):
    area_model = model if model in area_files else AREA_ALIASES.get(model)
    if area_model is None or area_model not in area_files:
        raise FileNotFoundError(f"no areacello file found for {model}")
    return area_files[area_model]


def regrid_model(source_path, area_path, method, target):
    with ExitStack() as stack:
        source = stack.enter_context(open_dataset(source_path))
        source = select_first_steps(source, "tos", 1)
        area = stack.enter_context(open_dataset(area_path)) if area_path else None

        if method == "nearest":
            result = regrid_nearest(
                source,
                "tos",
                target,
                area_dataset=area,
            )
        elif method == "binned":
            result = regrid_area_weighted_bins(
                source,
                "tos",
                target,
                area,
            )["tos"]
        else:
            result = regridder(
                source,
                "tos",
                target,
                method="nearest_s2d",
                periodic=True,
            )

        if result.dims[-2:] != ("lat", "lon"):
            raise ValueError(f"unexpected regridded dimensions: {result.dims}")
        return result.sizes["lat"], result.sizes["lon"]


def main():
    args = parse_args()
    model_files = build_model_files(args.data_root, args.scenario)
    models = set.intersection(*(set(files) for files in model_files.values()))
    if args.models:
        models &= set(args.models)

    area_files = first_files_by_model(args.data_root / "areacello", "areacello")
    target = standard_grid(args.resolution)

    for model in sorted(models):
        try:
            area_path = (
                None
                if args.method == "regridder"
                else find_area_file(model, area_files)
            )
            shape = regrid_model(
                model_files["tos"][model],
                area_path,
                args.method,
                target,
            )
            print(f"{model}: {shape[0]}x{shape[1]}")
        except Exception as exc:
            print(f"{model}: ERROR ({exc})")


if __name__ == "__main__":
    main()
