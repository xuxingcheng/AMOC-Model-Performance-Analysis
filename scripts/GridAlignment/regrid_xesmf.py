#!/usr/bin/env python
"""Regrid a CMIP field to standard lat/lon using xESMF."""

import argparse
from contextlib import ExitStack

from grids import (
    as_standard_dataset,
    describe_grid,
    open_dataset,
    regrid_with_xesmf,
    select_first_steps,
    standard_grid,
    summarize,
    write_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("variable")
    parser.add_argument("output")
    parser.add_argument(
        "--method",
        choices=(
            "bilinear",
            "conservative",
            "conservative_normed",
            "nearest_s2d",
            "nearest_d2s",
            "patch",
        ),
        default="bilinear",
    )
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--time-steps", type=int, default=1)
    parser.add_argument("--weights-file", default=None)
    parser.add_argument("--area-input", default=None)
    parser.add_argument("--area-variable", default="areacello")
    return parser.parse_args()


def main():
    args = parse_args()
    with ExitStack() as stack:
        source = stack.enter_context(open_dataset(args.input))
        area = stack.enter_context(open_dataset(args.area_input)) if args.area_input else None
        source = select_first_steps(source, args.variable, args.time_steps)
        target = standard_grid(args.resolution)
        result = regrid_with_xesmf(
            source,
            args.variable,
            target,
            method=args.method,
            weights_file=args.weights_file,
            area_dataset=area,
            area_variable=args.area_variable,
        )
        write_dataset(as_standard_dataset(result, target, args.variable), args.output)
        print("source:", describe_grid(source, args.variable))
        print("output:", summarize(result))


if __name__ == "__main__":
    main()
