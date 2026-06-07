#!/usr/bin/env python
"""Regrid a CMIP field by source-area-weighted assignment to regular lat/lon bins."""

import argparse

import numpy as np

from grids import (
    as_standard_dataset,
    describe_grid,
    open_dataset,
    regrid_area_weighted_bins,
    select_first_steps,
    source_valid_area_totals,
    standard_grid,
    summarize,
    write_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("variable")
    parser.add_argument("area_input")
    parser.add_argument("output")
    parser.add_argument("--area-variable", default="areacello")
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--time-steps", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    with open_dataset(args.input) as source, open_dataset(args.area_input) as area:
        source = select_first_steps(source, args.variable, args.time_steps)
        target = standard_grid(args.resolution)
        result = regrid_area_weighted_bins(
            source,
            args.variable,
            target,
            area,
            area_variable=args.area_variable,
        )
        source_area = source_valid_area_totals(
            source,
            args.variable,
            area,
            area_variable=args.area_variable,
        )
        write_dataset(as_standard_dataset(result, target), args.output)
        print("source:", describe_grid(source, args.variable))
        print("output:", summarize(result[args.variable]))
        assigned_area = result["source_area_sum"].sum(("lat", "lon")).values
        relative_error = np.abs(assigned_area.ravel() - source_area) / source_area
        print("assigned source area:", assigned_area)
        print("maximum area conservation relative error:", float(np.nanmax(relative_error)))


if __name__ == "__main__":
    main()
