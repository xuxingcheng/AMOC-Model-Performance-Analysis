import gc
import os
import pickle
import warnings

import dask
import gsw
import intake
import numpy as np
import pandas as pd
import xarray as xr

warnings.simplefilter("ignore", category=xr.SerializationWarning)

URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
EXPERIMENT_ID = "piControl"
MEMBER_ID = "r1i1p1f1"

LAST_N_YEAR = 20
LAST_N_MONTHS = 2
STEP_SIZE = 0.5
RHO_MIN = 1015
RHO_MAX = 1030
RHO_CLASSES = np.arange(RHO_MIN - STEP_SIZE, RHO_MAX + STEP_SIZE, STEP_SIZE)

SAVE_DIR = "/glade/work/stevenxu/AMOC_models"
SAVE_STEM = "Fgen_Allmodels"
SAVE_EVERY_N_MODELS = 3


def subtract_years_cftime(t, years):
    """Return a new cftime object with years subtracted (same month/day)."""
    if not hasattr(t, "year"):
        raise TypeError(f"Unsupported time scalar type: {type(t)}")
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


def get_source_ids(col, variable_id):
    query = col.search(
        experiment_id=EXPERIMENT_ID,
        variable_id=variable_id,
        member_id=MEMBER_ID,
    )
    df = query.df
    if df.empty:
        return set()
    return set(df["source_id"].unique())


def load_one_dataset(col, model, variable_id):
    query = col.search(
        source_id=[model],
        experiment_id=EXPERIMENT_ID,
        variable_id=variable_id,
        member_id=MEMBER_ID,
    )
    dset_dict = query.to_dataset_dict(
        zarr_kwargs={"consolidated": True},
        xarray_combine_by_coords_kwargs={"compat": "override", "join": "override"},
    )
    if not dset_dict:
        return None
    datasets = list(dset_dict.values())
    selected = datasets[-1]
    for ds in datasets[:-1]:
        try:
            ds.close()
        except Exception:
            pass
    return selected


def load_model_inputs(col, model):
    var_map = {
        "tos": "tos",
        "sos": "sos",
        "hfds": "hfds",
        "wfo": "wfo",
    }

    out = {}
    for short_name, var_id in var_map.items():
        ds = load_one_dataset(col, model, var_id)
        if ds is None:
            return None, f"missing {var_id}"
        out[short_name] = ds
    return out, None


def align_and_trim_inputs(model, ds_map):
    end_times = [
        ds_map["tos"]["time"].isel(time=-1).values.item(),
        ds_map["sos"]["time"].isel(time=-1).values.item(),
        ds_map["hfds"]["time"].isel(time=-1).values.item(),
        ds_map["wfo"]["time"].isel(time=-1).values.item(),
    ]

    min_end_time = min(end_times)
    start_time = subtract_years_cftime(min_end_time, LAST_N_YEAR)
    end_time = min_end_time

    print(
        f"Aligning time for {model} to last {LAST_N_YEAR} years: "
        f"{start_time} to {end_time}"
    )

    for k in list(ds_map):
        ds_map[k] = (
            ds_map[k]
            .sel(time=slice(start_time, end_time))
            .isel(time=slice(-LAST_N_MONTHS, None))
            .load()
        )

    return ds_map


def compute_surface_density(ds_tos, ds_sos):
    T, SP = xr.align(ds_tos["tos"], ds_sos["sos"], join="inner")
    p0 = 0.0

    alpha = xr.apply_ufunc(
        gsw.alpha,
        SP,
        T,
        p0,
        dask="parallelized",
        vectorize=True,
        output_dtypes=[float],
    )

    beta = xr.apply_ufunc(
        gsw.beta,
        SP,
        T,
        p0,
        dask="parallelized",
        vectorize=True,
        output_dtypes=[float],
    )

    rho = xr.apply_ufunc(
        gsw.rho,
        SP,
        T,
        p0,
        dask="parallelized",
        vectorize=True,
        output_dtypes=[float],
    )

    return {"rho": rho, "alpha": alpha, "beta": beta}


def compute_fsurf(ds_map):
    cp = 4.09e3
    rho0 = 1026
    rho_fw = 1000
    S0 = 35

    HF, WF = xr.align(ds_map["hfds"]["hfds"], ds_map["wfo"]["wfo"], join="inner")

    density = compute_surface_density(ds_map["tos"], ds_map["sos"])
    rho = density["rho"]
    alpha = density["alpha"]
    beta = density["beta"]

    HF, WF, rho, alpha, beta = xr.align(HF, WF, rho, alpha, beta, join="inner")
    fsurf = (alpha / cp) * HF + (rho0 / rho_fw) * beta * S0 * WF
    heat_comp = (alpha / cp) * HF
    fw_comp = (rho0 / rho_fw) * beta * S0 * WF

    return xr.Dataset(dict(fsurf=fsurf, rho=rho, heat_comp=heat_comp, fw_comp=fw_comp))


def load_areacello(col, model):
    ds = load_one_dataset(col, model, "areacello")
    if ds is None or "areacello" not in ds:
        return None
    area = ds["areacello"]
    if "time" in area.dims:
        area = area.isel(time=0).squeeze()
    else:
        area = area.squeeze()
    area = area.load()
    try:
        ds.close()
    except Exception:
        pass
    return area


def get_latitude_mask(Fsurf_data, dim_a=None, dim_b=None):
    if "latitude" in Fsurf_data:
        return Fsurf_data["latitude"] > 45
    if "lat" in Fsurf_data.coords:
        lat = Fsurf_data["lat"]
        if dim_a is None:
            return lat > 45
        mask = lat > 45
        if dim_b is None:
            return mask
        # Ensure the mask has both spatial dims before stacking.
        if set(mask.dims) != {dim_a, dim_b}:
            mask, _ = xr.broadcast(mask, Fsurf_data["fsurf"].isel(time=0))
        return mask
    raise KeyError("No latitude/lat field available for latitude filtering.")


def finalize_rows(rows):
    if not rows:
        return pd.DataFrame(
            columns=["rho", "Fgen", "HeatFlux", "FreshwaterFlux", "AreaSum"]
        )
    df = pd.DataFrame(
        rows,
        columns=["time", "rho", "Fgen", "HeatFlux", "FreshwaterFlux", "AreaSum"],
    )
    return df.groupby("rho", as_index=False)[
        ["Fgen", "HeatFlux", "FreshwaterFlux", "AreaSum"]
    ].mean()


def integrate_i(Fsurf_data, area1):
    filtered = Fsurf_data.where(get_latitude_mask(Fsurf_data) > 0, drop=True).isel(time=slice(0, 2))
    fsurf = filtered["fsurf"]
    heat_comp = filtered["heat_comp"]
    fw_comp = filtered["fw_comp"]
    rho = filtered["rho"]
    area1 = area1.sel(i=filtered["i"])

    weighted_fsurf = fsurf * area1
    weighted_heat_comp = heat_comp * area1
    weighted_fw_comp = fw_comp * area1

    rho_np, wf_np, wh_np, wfw_np, area_v = dask.compute(
        rho.data,
        weighted_fsurf.data,
        weighted_heat_comp.data,
        weighted_fw_comp.data,
        area1.data,
    )

    area_v = area1.values
    n_time = min(2, rho_np.shape[0], wf_np.shape[0], wh_np.shape[0], wfw_np.shape[0])
    rows = []
    for t in range(n_time):
        rho_t = rho_np[t]
        wf_t = wf_np[t]
        wh_t = wh_np[t]
        wfw_t = wfw_np[t]

        for rhoclass in RHO_CLASSES:
            rhobot = rhoclass
            rhotop = rhoclass + STEP_SIZE
            idx = np.where((rho_t > rhobot) & (rho_t < rhotop))[0]

            rows.append(
                [
                    t,
                    rhoclass + STEP_SIZE / 2,
                    float(np.nansum(wf_t[idx]) / STEP_SIZE / 1e6),
                    float(np.nansum(wh_t[idx]) / STEP_SIZE / 1e6),
                    float(np.nansum(wfw_t[idx]) / STEP_SIZE / 1e6),
                    float(np.nansum(area_v[idx])),
                ]
            )

    return finalize_rows(rows)


def integrate_ij_or_xy(Fsurf_data, area1, dim_a, dim_b):
    mask = get_latitude_mask(Fsurf_data, dim_a, dim_b)
    keep_pts = mask.stack(points=(dim_a, dim_b)).where(mask.stack(points=(dim_a, dim_b)), drop=True).coords["points"]

    fsurf_s = Fsurf_data["fsurf"].stack(points=(dim_a, dim_b)).sel(points=keep_pts).isel(time=slice(0, 2))
    heat_s = Fsurf_data["heat_comp"].stack(points=(dim_a, dim_b)).sel(points=keep_pts).isel(time=slice(0, 2))
    fw_s = Fsurf_data["fw_comp"].stack(points=(dim_a, dim_b)).sel(points=keep_pts).isel(time=slice(0, 2))
    rho_s = Fsurf_data["rho"].stack(points=(dim_a, dim_b)).sel(points=keep_pts).isel(time=slice(0, 2))
    area_s = area1.stack(points=(dim_a, dim_b)).sel(points=keep_pts)

    wf_s = fsurf_s * area_s
    wh_s = heat_s * area_s
    wfw_s = fw_s * area_s

    wf_np, wh_np, wfw_np, rho_np, area_np = dask.compute(
        wf_s.data,
        wh_s.data,
        wfw_s.data,
        rho_s.data,
        area_s.data,
    )

    rows = []
    n_time = min(2, rho_np.shape[0], wf_np.shape[0], wh_np.shape[0], wfw_np.shape[0])
    for t in range(n_time):
        rho_t = rho_np[t, :]

        for rhoclass in RHO_CLASSES:
            rhobot = rhoclass
            rhotop = rhoclass + STEP_SIZE
            idx = np.where((rho_t > rhobot) & (rho_t < rhotop))[0]

            rows.append(
                [
                    t,
                    rhoclass + STEP_SIZE / 2,
                    float(np.nansum(wf_np[t, idx]) / STEP_SIZE / 1e6),
                    float(np.nansum(wh_np[t, idx]) / STEP_SIZE / 1e6),
                    float(np.nansum(wfw_np[t, idx]) / STEP_SIZE / 1e6),
                    float(np.nansum(area_np[idx])),
                ]
            )

    return finalize_rows(rows)


def integrate_latlon(Fsurf_data, area1):
    filtered = Fsurf_data.where(Fsurf_data["lat"] > 45, drop=True).isel(time=slice(0, 2))

    # Handle 180x288 area against 90x144 fsurf if needed.
    if (
        area1.sizes.get("lat", 0) == filtered.sizes.get("lat", 0) * 2
        and area1.sizes.get("lon", 0) == filtered.sizes.get("lon", 0) * 2
    ):
        area1 = area1.coarsen(lat=2, lon=2, boundary="trim").sum()

    area1 = area1.sel(lat=filtered["lat"], lon=filtered["lon"], method="nearest")

    fsurf = filtered["fsurf"]
    heat_comp = filtered["heat_comp"]
    fw_comp = filtered["fw_comp"]
    rho = filtered["rho"]

    rho_np, fsurf_np, heat_np, fw_np, area_v = dask.compute(
        rho.data,
        fsurf.data,
        heat_comp.data,
        fw_comp.data,
        area1.data,
    )

    area_v = area1.values
    n_time = min(2, rho_np.shape[0], fsurf_np.shape[0], heat_np.shape[0], fw_np.shape[0])
    rows = []
    for t in range(n_time):
        rho_t = rho_np[t]
        fsurf_t = fsurf_np[t]
        heat_t = heat_np[t]
        fw_t = fw_np[t]
        wf_t = fsurf_t * area_v
        wh_t = heat_t * area_v
        wfw_t = fw_t * area_v

        for rhoclass in RHO_CLASSES:
            rhobot = rhoclass
            rhotop = rhoclass + STEP_SIZE
            m = (rho_t > rhobot) & (rho_t < rhotop)

            rows.append(
                [
                    t,
                    rhoclass + STEP_SIZE / 2,
                    float(np.nansum(wf_t[m]) / STEP_SIZE / 1e6),
                    float(np.nansum(wh_t[m]) / STEP_SIZE / 1e6),
                    float(np.nansum(wfw_t[m]) / STEP_SIZE / 1e6),
                    float(np.nansum(area_v[m])),
                ]
            )

    return finalize_rows(rows)


def compute_fgen_for_model(model, Fsurf_data, area_data):
    dims = set(Fsurf_data.dims)

    if {"i", "j"}.issubset(dims):
        return integrate_ij_or_xy(Fsurf_data, area_data, "j", "i")
    if {"x", "y"}.issubset(dims):
        return integrate_ij_or_xy(Fsurf_data, area_data, "y", "x")
    if "i" in dims and "j" not in dims:
        return integrate_i(Fsurf_data, area_data)
    if {"lat", "lon"}.issubset(dims):
        return integrate_latlon(Fsurf_data, area_data)

    print(f"Skipping {model}: unsupported grid dims {dims}")
    return None


def dump_batch(fgen_dict, batch_idx):
    save_path = os.path.join(SAVE_DIR, f"{SAVE_STEM}{batch_idx}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(fgen_dict, f)
    print(f"Saved batch {batch_idx}: {len(fgen_dict)} models -> {save_path}")
    return save_path


def main():
    col = intake.open_esm_datastore(URL)

    print("Discovering models by variable availability...")
    tos_models = get_source_ids(col, "tos")
    sos_models = get_source_ids(col, "sos")
    hfds_models = get_source_ids(col, "hfds")
    wfo_models = get_source_ids(col, "wfo")
    area_models = get_source_ids(col, "areacello")

    core_models = tos_models & sos_models & hfds_models & wfo_models
    models = sorted(core_models & area_models)
    print(f"Candidate models with tos/sos/hfds/wfo: {len(core_models)}")
    print(f"Candidate models with tos/sos/hfds/wfo/areacello: {len(models)}")

    Fgen_dict = {}
    batch_idx = 1
    total_saved_models = 0

    for model in models:
        print(f"\n=== Processing {model} ===")
        ds_map = None
        fsurf_ds = None
        area_ds = None

        try:
            ds_map, reason = load_model_inputs(col, model)
            if ds_map is None:
                print(f"Skipping {model}: {reason}")
                continue

            ds_map = align_and_trim_inputs(model, ds_map)
            fsurf_ds = compute_fsurf(ds_map)

            area_ds = load_areacello(col, model)
            if area_ds is None:
                print(f"Skipping {model}: missing areacello")
                continue

            fgen = compute_fgen_for_model(model, fsurf_ds, area_ds)
            if fgen is None or fgen.empty:
                print(f"Skipping {model}: no Fgen rows produced")
                continue

            Fgen_dict[model] = fgen
            print(f"Completed Fgen for {model} with {len(fgen)} rho bins")
            if len(Fgen_dict) >= SAVE_EVERY_N_MODELS:
                dump_batch(Fgen_dict, batch_idx)
                total_saved_models += len(Fgen_dict)
                Fgen_dict.clear()
                gc.collect()
                batch_idx += 1

        except Exception as e:
            print(f"⚠ Error processing {model}: {repr(e)}")

        finally:
            if ds_map is not None:
                for ds in ds_map.values():
                    try:
                        ds.close()
                    except Exception:
                        pass
            del ds_map, fsurf_ds, area_ds
            gc.collect()

    if Fgen_dict:
        dump_batch(Fgen_dict, batch_idx)
        total_saved_models += len(Fgen_dict)
        Fgen_dict.clear()
        gc.collect()

    print(f"Completed run. Total saved models across batches: {total_saved_models}")


if __name__ == "__main__":
    # Split large chunks aggressively during slicing/reshape to reduce peak memory.
    with dask.config.set({"array.slicing.split_large_chunks": True}):
        main()
