#!/usr/bin/env python
"""Regrid a CMIP field with the notebook-style xESMF Regridder."""

import argparse

from grids import (
    as_standard_dataset,
    describe_grid,
    open_dataset,
    regridder,
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
        default="nearest_s2d",
    )
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--time-steps", type=int, default=1)
    parser.add_argument("--weights-file", default=None)
    parser.add_argument("--no-periodic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with open_dataset(args.input) as source:
        source = select_first_steps(source, args.variable, args.time_steps)
        target = standard_grid(args.resolution)
        result = regridder(
            source,
            args.variable,
            target,
            method=args.method,
            periodic=not args.no_periodic,
            weights_file=args.weights_file,
        )
        write_dataset(as_standard_dataset(result, target, args.variable), args.output)
        print("source:", describe_grid(source, args.variable))
        print("output:", summarize(result))


if __name__ == "__main__":
    main()
