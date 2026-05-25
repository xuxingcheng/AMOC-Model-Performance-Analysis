import argparse
import gc
import os

import dask
import numpy as np
import pandas as pd
import xarray as xr

import Fgenrun2_streaming as fgen


DEFAULT_OUTPUT = os.path.join(
    fgen.DATA_ROOT,
    "AverageHeatFlux_50_70N_Allmodels_streaming.pkl",
)
DEFAULT_LAT_MIN = 50.0
DEFAULT_LAT_MAX = 70.0
DEFAULT_LON_MIN = -90.0
DEFAULT_LON_MAX = 60.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sequential area-weighted average raw heat flux driver. For each "
            "model, it averages hfds over the selected region using areacello "
            "weights and requiring finite fsurf/rho validity."
        )
    )
    parser.add_argument("--scenario", default=fgen.DEFAULT_SCENARIO)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--last-n-years", type=int, default=fgen.DEFAULT_LAST_N_YEARS)
    parser.add_argument("--last-n-months", type=int, default=fgen.DEFAULT_LAST_N_MONTHS)
    parser.add_argument("--time-chunk", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=sorted(fgen.DEFAULT_EXCLUDING_MODELS),
    )
    return parser.parse_args()


def get_region_mask(fsurf_ds, spatial_dims, lat_min, lat_max, lon_min, lon_max):
    sample = fsurf_ds["fsurf"].isel(time=0)

    if "latitude" in fsurf_ds:
        latitude = fsurf_ds["latitude"]
        if "longitude" not in fsurf_ds:
            raise KeyError("No longitude field available for filtering.")
        longitude = fgen.normalize_longitude(fsurf_ds["longitude"])
        if set(latitude.dims) != set(spatial_dims):
            latitude, _ = xr.broadcast(latitude, sample)
        if set(longitude.dims) != set(spatial_dims):
            longitude, _ = xr.broadcast(longitude, sample)
        return (
            (latitude >= lat_min)
            & (latitude <= lat_max)
            & (longitude >= lon_min)
            & (longitude <= lon_max)
        )

    if "lat" in fsurf_ds.coords:
        latitude = fsurf_ds["lat"]
        if "lon" not in fsurf_ds.coords:
            raise KeyError("No lon coordinate available for filtering.")
        longitude = fgen.normalize_longitude(fsurf_ds["lon"])
        mask = (
            (latitude >= lat_min)
            & (latitude <= lat_max)
            & (longitude >= lon_min)
            & (longitude <= lon_max)
        )
        if set(mask.dims) != set(spatial_dims):
            mask, _ = xr.broadcast(mask, sample)
        return mask

    raise KeyError("No latitude/longitude coordinates available for filtering.")


def compute_average_heatflux_for_model(
    model,
    fsurf_ds,
    raw_heat_da,
    area_da,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
):
    spatial_dims = fgen.get_spatial_dims(fsurf_ds)
    if spatial_dims is None:
        print(f"Skipping {model}: unsupported spatial dims {fsurf_ds['fsurf'].dims}")
        return None

    fgen.assert_nonempty_spatial_grid(
        model,
        fsurf_ds["fsurf"],
        "fsurf before average heat flux",
    )
    raw_heat_da, fsurf_da, rho_da = xr.align(
        raw_heat_da,
        fsurf_ds["fsurf"],
        fsurf_ds["rho"],
        join="exact",
    )
    area_ready = fgen.prepare_area_for_fsurf(area_da, fsurf_ds, spatial_dims)
    mask = get_region_mask(
        fsurf_ds=fsurf_ds,
        spatial_dims=spatial_dims,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )
    mask_s = mask.stack(points=spatial_dims)
    keep_pts = mask_s.where(mask_s, drop=True).coords["points"]
    if keep_pts.size == 0:
        print(f"Skipping {model}: no points in the target lat/lon range.")
        return None

    raw_heat_s = raw_heat_da.stack(points=spatial_dims).sel(points=keep_pts)
    fsurf_s = fsurf_da.stack(points=spatial_dims).sel(points=keep_pts)
    rho_s = rho_da.stack(points=spatial_dims).sel(points=keep_pts)
    area_s = area_ready.stack(points=spatial_dims).sel(points=keep_pts)
    time_values = raw_heat_s["time"].values

    heat_np, fsurf_np, rho_np, area_np = dask.compute(
        raw_heat_s.data,
        fsurf_s.data,
        rho_s.data,
        area_s.data,
    )

    area_flat = np.ravel(area_np)
    finite_positive_area = np.isfinite(area_flat) & (area_flat > 0)

    rows = []
    for time_index, time_value in enumerate(time_values):
        heat_flat = np.ravel(heat_np[time_index])
        fsurf_flat = np.ravel(fsurf_np[time_index])
        rho_flat = np.ravel(rho_np[time_index])
        valid = (
            finite_positive_area
            & np.isfinite(heat_flat)
            & np.isfinite(fsurf_flat)
            & np.isfinite(rho_flat)
        )

        # Calculation: sum(raw hfds * areacello) / sum(areacello) over the selected lat/lon
        area_sum = float(np.nansum(area_flat[valid]))
        weighted_heatflux_sum = float(np.nansum(heat_flat[valid] * area_flat[valid]))
        if area_sum > 0:
            average_heatflux = weighted_heatflux_sum / area_sum
        else:
            average_heatflux = np.nan

        rows.append(
            [
                time_index,
                time_value,
                float(average_heatflux),
                area_sum,
                weighted_heatflux_sum,
                int(np.count_nonzero(valid)),
            ]
        )

    average_heatflux = pd.DataFrame(
        rows,
        columns=[
            "time_index",
            "time",
            "AverageHeatFlux",
            "AreaSum",
            "WeightedHeatFluxSum",
            "n_points",
        ],
    )
    average_heatflux.attrs["heat_flux_variable"] = "hfds"
    average_heatflux.attrs["heat_flux_source"] = "raw CMIP hfds"
    average_heatflux.attrs["heat_flux_units"] = raw_heat_da.attrs.get("units", "W m-2")
    average_heatflux.attrs["area_units"] = area_da.attrs.get("units", "m2")
    average_heatflux.attrs["lat_min"] = lat_min
    average_heatflux.attrs["lat_max"] = lat_max
    average_heatflux.attrs["lon_min"] = lon_min
    average_heatflux.attrs["lon_max"] = lon_max
    average_heatflux.attrs["longitude_range"] = "-180 to 180"
    average_heatflux.attrs["fsurf_validity_filter"] = "finite fsurf and finite rho"
    average_heatflux.attrs["description"] = (
        "AverageHeatFlux is sum(raw hfds * areacello) / sum(areacello) over "
        "the selected latitude/longitude bounds, requiring finite raw hfds, "
        "finite fsurf, finite rho, and finite positive areacello."
    )
    return average_heatflux


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
    if args.lat_min > args.lat_max:
        raise ValueError("--lat-min must be less than or equal to --lat-max")
    if args.lon_min > args.lon_max:
        raise ValueError("--lon-min must be less than or equal to --lon-max")

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
    print(
        "Region: "
        f"{args.lat_min:g}-{args.lat_max:g}N, "
        f"{args.lon_min:g} to {args.lon_max:g} lon"
    )
    print("Heat flux: raw hfds, area-weighted by areacello")
    print("Validity filter: finite raw hfds, finite fsurf, finite rho, positive area")
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
            average_heatflux = compute_average_heatflux_for_model(
                model=model,
                fsurf_ds=fsurf_ds,
                raw_heat_da=trimmed_ds_map["hfds"]["hfds"],
                area_da=area_da,
                lat_min=args.lat_min,
                lat_max=args.lat_max,
                lon_min=args.lon_min,
                lon_max=args.lon_max,
            )
            if average_heatflux is None or average_heatflux.empty:
                print(f"Skipping {model}: no average heat flux rows produced.")
                continue

            results[model] = average_heatflux
            unsaved += 1
            mean_heatflux = average_heatflux["AverageHeatFlux"].mean()
            mean_area = average_heatflux["AreaSum"].mean()
            print(
                f"Completed {model}: {len(average_heatflux)} time steps, "
                f"mean AverageHeatFlux {mean_heatflux:.6e}, "
                f"mean AreaSum {mean_area:.6e}"
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
