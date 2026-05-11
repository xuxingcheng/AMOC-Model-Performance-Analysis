import argparse
import gc
import os

import dask
import numpy as np
import pandas as pd

import Fgenrun2_streaming as fgen


DEFAULT_OUTPUT = os.path.join(fgen.DATA_ROOT, "AreaSum_Allmodels_streaming.pkl")
DEFAULT_RHO_MIN = 1025.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sequential AreaSum driver. For each model, it sums areacello over "
            "grid cells where fsurf < 0, rho exceeds the threshold, and the "
            "existing North Atlantic spatial mask is true."
        )
    )
    parser.add_argument("--scenario", default=fgen.DEFAULT_SCENARIO)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--last-n-years", type=int, default=fgen.DEFAULT_LAST_N_YEARS)
    parser.add_argument("--last-n-months", type=int, default=fgen.DEFAULT_LAST_N_MONTHS)
    parser.add_argument("--rho-min", type=float, default=DEFAULT_RHO_MIN)
    parser.add_argument("--time-chunk", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=sorted(fgen.DEFAULT_EXCLUDING_MODELS),
    )
    return parser.parse_args()


def compute_areasum_for_model(model, fsurf_ds, area_da, rho_min):
    spatial_dims = fgen.get_spatial_dims(fsurf_ds)
    if spatial_dims is None:
        print(f"Skipping {model}: unsupported spatial dims {fsurf_ds['fsurf'].dims}")
        return None

    fgen.assert_nonempty_spatial_grid(model, fsurf_ds["fsurf"], "fsurf before AreaSum")
    area_ready = fgen.prepare_area_for_fsurf(area_da, fsurf_ds, spatial_dims)
    stacked = fgen.stack_north_of_45(model, fsurf_ds, area_ready, spatial_dims)
    if stacked is None:
        print(f"Skipping {model}: no points in the target lat/lon range.")
        return None

    rho_s, fsurf_s, _heat_s, _fw_s, area_s = stacked
    time_values = rho_s["time"].values

    rho_np, fsurf_np, area_np = dask.compute(
        rho_s.data,
        fsurf_s.data,
        area_s.data,
    )

    area_flat = np.ravel(area_np)
    finite_area = np.isfinite(area_flat)

    rows = []
    for time_index, time_value in enumerate(time_values):
        rho_flat = np.ravel(rho_np[time_index])
        fsurf_flat = np.ravel(fsurf_np[time_index])
        valid = (
            np.isfinite(fsurf_flat)
            & np.isfinite(rho_flat)
            & finite_area
            & (fsurf_flat < 0)
            & (rho_flat > rho_min)
        )
        rows.append(
            [
                time_index,
                time_value,
                float(np.nansum(area_flat[valid])),
            ]
        )

    areasum = pd.DataFrame(rows, columns=["time_index", "time", "AreaSum"])
    areasum.attrs["rho_min"] = rho_min
    areasum.attrs["description"] = (
        "AreaSum is the sum of areacello where fsurf < 0, rho > rho_min, "
        "and the existing North Atlantic mask is true."
    )
    return areasum


def get_time_chunk(args):
    if args.time_chunk is not None:
        return args.time_chunk
    if args.last_n_months is not None:
        return max(12, args.last_n_months)
    return max(12, args.last_n_years * 12)


def main():
    args = parse_args()
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")

    time_chunk = get_time_chunk(args)
    registry = fgen.build_model_file_registry(args.scenario)
    area_index = fgen.group_files_by_model(fgen.AREA_DIR)
    candidate_models = fgen.get_candidate_models(
        registry=registry,
        include_models=args.models,
        exclude_models=args.exclude_models,
    )

    if args.max_models is not None:
        candidate_models = candidate_models[: args.max_models]

    results = fgen.load_existing_results(args.output) if args.resume else {}
    processed_before = set(results)
    models = [model for model in candidate_models if model not in processed_before]

    print(f"Scenario: {args.scenario}")
    print(f"Candidate models with tos/sos/hfds/wfo: {len(candidate_models)}")
    print(f"Models remaining this run: {len(models)}")
    print(f"rho_min threshold: {args.rho_min}")
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
            area_da = fgen.load_area_for_model(model, area_index)
            if area_da is None:
                print(f"Skipping {model}: no areacello file or alias.")
                continue

            fsurf_ds = fgen.compute_fsurf(model, trimmed_ds_map, area_da).load()
            areasum = compute_areasum_for_model(
                model=model,
                fsurf_ds=fsurf_ds,
                area_da=area_da,
                rho_min=args.rho_min,
            )
            if areasum is None or areasum.empty:
                print(f"Skipping {model}: no AreaSum rows produced.")
                continue

            results[model] = areasum
            unsaved += 1
            mean_areasum = areasum["AreaSum"].mean()
            print(
                f"Completed {model}: {len(areasum)} time steps, "
                f"mean AreaSum {mean_areasum:.6e}"
            )

            if unsaved >= args.save_every:
                fgen.save_results(args.output, results)
                print(f"Checkpoint saved with {len(results)} models")
                unsaved = 0

        except Exception as exc:
            print(f"Error while processing {model}: {exc!r}")

        finally:
            if opened_ds_map is not None:
                for ds in opened_ds_map.values():
                    fgen.safe_close(ds)
            if trimmed_ds_map is not None:
                for ds in trimmed_ds_map.values():
                    fgen.safe_close(ds)
            fgen.safe_close(fsurf_ds)
            del opened_ds_map, backend_map, trimmed_ds_map, area_da, fsurf_ds
            gc.collect()

    if unsaved:
        fgen.save_results(args.output, results)
        print(f"Final checkpoint saved with {len(results)} models")

    print(f"Done. Total models in output: {len(results)}")


if __name__ == "__main__":
    with dask.config.set({"array.slicing.split_large_chunks": True}):
        main()
