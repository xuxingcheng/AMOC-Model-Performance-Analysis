# %%
import os
import numpy as np
import glob
from collections import defaultdict
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gsw
from pyproj import Geod
import pandas as pd
import warnings
import dask
import intake
import gc


# %% [markdown]
# # Data Import

# %%


# Suppress serialization warnings for deep-time calendars
warnings.simplefilter("ignore", category=xr.SerializationWarning)

# Connect to the catalog once globally
url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
#url = 'https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.json'
col = intake.open_esm_datastore(url)

target_models = ['MRI-ESM2-0']
models_found = [] # We will use this to build your 'models' set

def get_cloud_datasets(experiment, variable_id):
    """
    Fetches data from the cloud and returns a dict keyed by model name.
    Restricts to a single member_id to prevent merge conflicts.
    """
    print(f"Querying cloud catalog for {variable_id} in {experiment}...")
    
    query = col.search(
        #source_id=target_models,
        experiment_id=experiment,
        variable_id=variable_id,
        table_id="Omon",
        member_id="r1i1p1f1"  # <--- THIS IS THE FIX
    )
    
    # Load lazy datasets
    dset_dict = query.to_dataset_dict(
        zarr_kwargs={'consolidated': True},
        xarray_combine_by_coords_kwargs={'compat': 'override', 'join': 'override'}
    )
    
    clean_datasets = {}
    for full_key, ds in dset_dict.items():
        # Extract the model name from the key
        model_name = full_key.split('.')[2]
        
        clean_datasets[model_name] = ds
        models_found.append(model_name)
        
    print(f"  -> Successfully loaded {len(clean_datasets)} models for {variable_id}\n")
    return clean_datasets

sst_datasets = get_cloud_datasets("piControl", "tos")
sss_datasets = get_cloud_datasets("piControl", "sos")
hf_datasets  = get_cloud_datasets("piControl", "hfds")
wf_datasets  = get_cloud_datasets("piControl", "wfo")

models = set(models_found)
print("Final loaded models:", models)
all_loaded_models = set(models)

# %%
def subtract_years_cftime(t, years):
    """Return a new cftime object with years subtracted (same month/day)."""
    return type(t)(
        t.year - years, t.month, t.day,
        t.hour, t.minute, t.second, t.microsecond,
        has_year_zero=t.has_year_zero
    )

valid_models = set()
def align_time(model, lastNyear):
    # Make sure the model exists everywhere
    for name, dct in [
        ("sst", sst_datasets),
        ("sss", sss_datasets),
        ("hf",  hf_datasets),
        ("wf",  wf_datasets),
    ]:
        if model not in dct:
            print(f"Skipping {model}: missing in {name}_datasets")
            return

    try:
        end_times = []
        for variable_ds in [sst_datasets, sss_datasets, hf_datasets, wf_datasets]:
            end_times.append(variable_ds[model]['time'].isel(time=-1).values.item())

        # Calculate the shared time bounds
        min_end_time = min(end_times)
        start_time = subtract_years_cftime(min_end_time, lastNyear)
        end_time   = min_end_time
        
        print(f"Aligning time for {model} to last {lastNyear} years: {start_time} to {end_time}")

        # Slice the datasets
        sst_datasets[model] = sst_datasets[model].sel(time=slice(start_time, end_time))
        sss_datasets[model] = sss_datasets[model].sel(time=slice(start_time, end_time))
        hf_datasets[model]  = hf_datasets[model].sel(time=slice(start_time, end_time))
        wf_datasets[model]  = wf_datasets[model].sel(time=slice(start_time, end_time))

    except Exception as e:
        # Catch ANY error (slicing, calendar mismatches, missing dims) and skip gracefully
        print(f"⚠ Error aligning time for {model}. Skipping this model.")
        print(f"  -> Details: {repr(e)}\n")
        return

    valid_models.add(model)

# Apply the alignment
for model in list(models): # Use list(models) so we don't modify the set while iterating if you drop models later
    align_time(model, 20)

models = valid_models
unused_models = all_loaded_models - models
for dct in [sst_datasets, sss_datasets, hf_datasets, wf_datasets]:
    for model in unused_models:
        dct.pop(model, None)

print(f"Dropped {len(unused_models)} models after time alignment to save memory.")
gc.collect()

# %% [markdown]
# # Calculate sea surface density

# %%
def compute_surface_density(model, sst_datasets, sss_datasets, last_n_months=None):
    T  = sst_datasets[model]['tos']          
    SP = sss_datasets[model]['sos']          
    # addubg vertices for accurate area-weighted calculations for next step
    #vertices_latitude = sst_datasets[model]["vertices_latitude"]
    #vertices_longitude = sst_datasets[model]["vertices_longitude"]

    p0 = 0.0

    if last_n_months is not None:
        T  = T.isel(time=slice(-last_n_months, None))
        SP = SP.isel(time=slice(-last_n_months, None))

    """SA = xr.apply_ufunc(
        gsw.SA_from_SP, SP, p0, lon, lat,
        dask='parallelized', vectorize=True, output_dtypes=[float]
    )
    CT = xr.apply_ufunc(
        gsw.CT_from_t, SA, T, p0,
        dask='parallelized', vectorize=True, output_dtypes=[float]
    )"""
    alpha = xr.apply_ufunc(
        gsw.density.alpha, SP, T, p0,
        dask='parallelized', output_dtypes=[float]
    )
    beta = xr.apply_ufunc(
        gsw.density.beta, SP, T, p0,
        dask='parallelized', output_dtypes=[float]
    )
    rho = xr.apply_ufunc(
        gsw.density.rho, SP, T, p0,
        dask='parallelized', vectorize=True, output_dtypes=[float]
    )

    rho  = rho.rename('rho').assign_attrs(units='kg m-3',  long_name='Sea-surface density')
    alpha  = alpha.rename('alpha')
    beta  = beta.rename('beta')

    return xr.Dataset(dict(rho=rho, alpha=alpha, beta=beta))

#surf_den_ACCESS = compute_surface_density("NorESM2-LM", sst_datasets, sss_datasets, last_n_months=240)
#surf_den_ACCESS


# %% [markdown]
# # Calculate F surf

# %%
def compute_fsurf(model,
                  sst_datasets, sss_datasets, hf_datasets, wf_datasets,
                  cp=3990.0, rho0=1027.0, rho_fw=1000.0, S0=35.0,
                  last_n_months=None):

    HF = hf_datasets[model]['hfds']  # W m^-2, 
    WF = wf_datasets[model]['wfo']     # kg m^-2 s^-1, 

    density_data = compute_surface_density(model, sst_datasets, sss_datasets, last_n_months=last_n_months)
    rho = density_data['rho']
    alpha = density_data['alpha']
    beta = density_data['beta']

    if last_n_months is not None:
        HF  = HF.isel(time=slice(-last_n_months, None))
        WF = WF.isel(time=slice(-last_n_months, None))

    # f_surf = -(alpha/cp) * f_heat  - (rho0/rho_fw) * beta * S0 * f_water
    fsurf = (alpha / cp) * HF  +  (rho0 / rho_fw) * beta * S0 * WF
    fsurf = fsurf.assign_attrs(
        long_name="Buoyancy-relevant surface forcing (Eq. 5)",
        description="(alpha/cp)*f_heat + (rho0/rho_fw)*beta*S0*f_water",
        units="",
        cp=cp, rho0=rho0, rho_fw=rho_fw, S0=S0
    )

    heat_comp = (alpha / cp) * HF
    fw_comp = (rho0 / rho_fw) * beta * S0 * WF

    return xr.Dataset(dict(fsurf=fsurf, rho=rho, heat_comp=heat_comp, fw_comp=fw_comp))

Fsurf_data = compute_fsurf(
    "CanESM5",
    sst_datasets=sst_datasets,
    sss_datasets=sss_datasets,
    hf_datasets=hf_datasets,
    wf_datasets=wf_datasets,
    last_n_months=240
)

Fsurf_datasets = {}
for model in models:
    print(f"Computing F_surf for {model}...")
    Fsurf_datasets[model] = compute_fsurf(
        model,
        sst_datasets=sst_datasets,
        sss_datasets=sss_datasets,
        hf_datasets=hf_datasets,
        wf_datasets=wf_datasets,
         last_n_months=240
    )


# %% [markdown]
# # Calculating Fgen at single timepoint

# %% [markdown]
# ### Create density class list
# Take max and min in rho values, and slice with interval 0.01

# %%
rho_min = 1015
rho_max = 1030


# %%
step_size = 0.05
rho_classes = np.arange(rho_min - step_size, rho_max + step_size, step_size)


# %% [markdown]
# ### Dataset for area at each grid cell

# %%
print("Querying cloud catalog for areacello...")

# Search for areacello, but only for the models that passed our alignment check
area_query = col.search(
    source_id=list(models),       # Use the valid models we just cleaned up
    experiment_id="piControl",
    variable_id="areacello",
    member_id="r1i1p1f1"
)

# Load the lazy datasets
area_dset_dict = area_query.to_dataset_dict(
    zarr_kwargs={'consolidated': True},
    xarray_combine_by_coords_kwargs={'compat': 'override', 'join': 'override'}
)

area_ds = {}

for full_key, ds in area_dset_dict.items():
    model_name = full_key.split('.')[2]
    
    # Extract just the DataArray to match your original script's behavior
    # and take the first time step if it accidentally has a time dimension
    if 'time' in ds['areacello'].dims:
        area = ds['areacello'].isel(time=0).squeeze()
    else:
        area = ds['areacello'].squeeze()
        
    area_ds[model_name] = area

# Check if any models are missing their area files
validated_models = []
for model in models:
    if model not in area_ds:
        print(f"⚠ Missing areacello for {model} in the cloud catalog!")
    else:
        validated_models.append(model)

models = set(validated_models)  # Update our models set to only include those with area data
for model in list(area_ds):
    if model not in models:
        area_ds.pop(model, None)
for model in list(Fsurf_datasets):
    if model not in models:
        Fsurf_datasets.pop(model, None)

# Raw cloud inputs are no longer needed after F_surf is built.
del sst_datasets, sss_datasets, hf_datasets, wf_datasets
gc.collect()

print(f"\nSuccessfully loaded areacello for {len(area_ds)} models.")

# %%
ij_models = []
xy_models = []
i_models = []
norm_models = []

for model_name, ds in Fsurf_datasets.items():
    if model_name not in models:
        print(f"Skipping {model_name} for dimension classification: missing from validated models.")
        continue
    dims = set(ds.dims)

    if {'i', 'j'}.issubset(dims):
        ij_models.append(model_name)
    elif {'x', 'y'}.issubset(dims):
        xy_models.append(model_name)
    elif 'i' in dims and 'j' not in dims:
        i_models.append(model_name)
    elif {'lat', 'lon'}.issubset(dims):
        norm_models.append(model_name)


# %%
print('ij_models:', ij_models)
print('xy_models:', xy_models)
print('i_models:', i_models)
print('norm_models:', norm_models)


# %% [markdown]
# Align areacello and actual data (180,288 -> 90, 144)

# %%
for model in norm_models:

    fs = Fsurf_datasets[model]
    area0 = area_ds[model]          # (lat:180, lon:288)

    # 1) coarsen to (lat:90, lon:144) — use SUM for area
    area_coarse = area0.coarsen(lat=2, lon=2, boundary="trim").sum()

    # 2) align to Fsurf grid (safe even if tiny coord diffs)
    area_coarse = area_coarse.sel(
        lat=fs["lat"],
        lon=fs["lon"],
        method="nearest"
    )

    # optional: sanity check
    print(model + ": Fsurf grid:", fs.sizes["lat"], fs.sizes["lon"])
    print(model + ": Area  grid:", area_coarse.sizes["lat"], area_coarse.sizes["lon"])

    # 3) save it for later use
    area_ds[model] = area_coarse


# %% [markdown]
# ### Integration

# %% [markdown]
# Group by density intervals and adding up the area-weighted fsurf

# %%
Fgen_dict = {}

# %% [markdown]
# #### (i) models

# %%
for model, Fsurf_data in Fsurf_datasets.items():
    if model not in i_models:
        continue

    filtered_Fsurf = (
        Fsurf_data
        .where(Fsurf_data["latitude"] > 45, drop=True)
    )#.isel(time=slice(9, 10))

    fsurf = filtered_Fsurf["fsurf"]
    heat_comp = filtered_Fsurf["heat_comp"]
    fw_comp = filtered_Fsurf["fw_comp"]
    rho = filtered_Fsurf["rho"]

    area1 = area_ds[model].sel(i=filtered_Fsurf["i"])  # <-- critical alignment

    timepoints = filtered_Fsurf["time"].values

    weighted_fsurf = fsurf * area1
    weighted_heat_comp = heat_comp * area1
    weighted_fw_comp = fw_comp * area1

    rho_np, wf_np, wh_np, wfw_np, area_v = dask.compute(
        rho.data,                 # (time, i)
        weighted_fsurf.data,      # (time, i)
        weighted_heat_comp.data,  # (time, i)
        weighted_fw_comp.data,    # (time, i)
        area1.data,               # (i,)
    )

    # numpy area
    area_v = area1.values

    rows = []  # faster than Fgen.loc in loop
    for time in range(len(timepoints)):
        # pull numpy once per time (avoids dask/xarray objects in pandas)
        rho_t = rho_np[time]
        wf_t  = wf_np[time]
        wh_t  = wh_np[time]
        wfw_t = wfw_np[time]

        for rhoclass in rho_classes:
            rhobot = rhoclass
            rhotop = rhoclass + step_size

            idx = np.where((rho_t > rhobot) & (rho_t < rhotop))[0]

            fgen_value = float(np.nansum(wf_t[idx])  / step_size / 1e6)
            heat_value = float(np.nansum(wh_t[idx])  / step_size / 1e6)
            fw_value   = float(np.nansum(wfw_t[idx]) / step_size / 1e6)
            area_sum   = float(np.nansum(area_v[idx]))

            rows.append([time, rhoclass + step_size/2, fgen_value, heat_value, fw_value, area_sum])

    Fgen = pd.DataFrame(rows, columns=["time", "rho", "Fgen", "HeatFlux", "FreshwaterFlux", "AreaSum"])
    Fgen = Fgen.groupby("rho", as_index=False)[["Fgen", "HeatFlux", "FreshwaterFlux", "AreaSum"]].mean()

    Fgen_dict[model] = Fgen
    print(f"Completed Fgen calculation for {model}")


# %% [markdown]
# #### (i, j) models

# %%
for model, Fsurf_data in Fsurf_datasets.items():
    if model not in ij_models:
        continue

    # -------------------------
    # 1) Build mask (j,i) and get kept points (MultiIndex labels)
    # -------------------------
    mask = (Fsurf_data["latitude"] > 45)

    mask_s = mask.stack(points=("j", "i"))  # MultiIndex points=(j,i)
    keep_pts = mask_s.where(mask_s, drop=True).coords["points"]

    # -------------------------
    # 2) Stack fields and select kept points (still lazy xarray)
    # -------------------------
    fsurf_s = Fsurf_data["fsurf"].stack(points=("j", "i")).sel(points=keep_pts)
    heat_s  = Fsurf_data["heat_comp"].stack(points=("j", "i")).sel(points=keep_pts)
    fw_s    = Fsurf_data["fw_comp"].stack(points=("j", "i")).sel(points=keep_pts)
    rho_s   = Fsurf_data["rho"].stack(points=("j", "i")).sel(points=keep_pts)

    # areacello (pick the first one if your area_ds stores a list per model)
    area1 = area_ds[model]
    area_s = area1.stack(points=("j", "i")).sel(points=keep_pts)

    # -------------------------
    # 3) Apply weights (still xarray), then convert to NumPy once
    #    Keep as (time, points) for speed (no need to unstack!)
    # -------------------------
    wf_s  = (fsurf_s * area_s)
    wh_s  = (heat_s  * area_s)
    wfw_s = (fw_s    * area_s)

    # OPTIONAL: restrict time for testing (matches your i-model testing)
    """wf_s  = wf_s.isel(time=slice(0, 1))
    wh_s  = wh_s.isel(time=slice(0, 1))
    wfw_s = wfw_s.isel(time=slice(0, 1))
    rho_s = rho_s.isel(time=slice(0, 1))"""

    timepoints = rho_s["time"].values

    # Convert to numpy arrays ONCE (prevents pandas/dask recursion issues)
    wf_np, wh_np, wfw_np, rho_np, area_np = dask.compute(
        wf_s.data,
        wh_s.data,
        wfw_s.data,
        rho_s.data,
        area_s.data,
    )

    # -------------------------
    # 4) Main loops (time x rho_bins), vectorized across points
    # -------------------------
    rows = []
    for t in range(len(timepoints)):
        rho_t = rho_np[t, :]

        for rhoclass in rho_classes:
            rhobot = rhoclass
            rhotop = rhoclass + step_size

            idx = np.where((rho_t > rhobot) & (rho_t < rhotop))[0]

            fgen_value  = float(np.nansum(wf_np[t, idx])  / step_size / 1e6)
            heat_value  = float(np.nansum(wh_np[t, idx])  / step_size / 1e6)
            fw_value    = float(np.nansum(wfw_np[t, idx]) / step_size / 1e6)
            area_sum    = float(np.nansum(area_np[idx]))

            rows.append([t, rhoclass + step_size/2, fgen_value, heat_value, fw_value, area_sum])

    Fgen = pd.DataFrame(rows, columns=["time", "rho", "Fgen", "HeatFlux", "FreshwaterFlux", "AreaSum"])
    Fgen = Fgen.groupby("rho", as_index=False)[["Fgen", "HeatFlux", "FreshwaterFlux", "AreaSum"]].mean()

    Fgen_dict[model] = Fgen
    print(f"Completed Fgen calculation for {model}")


# %% [markdown]
# #### (x, y) models

# %%
for model, Fsurf_data in Fsurf_datasets.items():
    if model not in xy_models: # 
        continue

    # -------------------------
    # 1) Build mask (y,x) and get kept points (MultiIndex labels)
    # -------------------------
    mask = (Fsurf_data["latitude"] > 45)

    mask_s = mask.stack(points=("y", "x"))  # MultiIndex points=(y,x)
    keep_pts = mask_s.where(mask_s, drop=True).coords["points"]

    # -------------------------
    # 2) Stack fields and select kept points (still lazy xarray)
    # -------------------------
    fsurf_s = Fsurf_data["fsurf"].stack(points=("y", "x")).sel(points=keep_pts)
    heat_s  = Fsurf_data["heat_comp"].stack(points=("y", "x")).sel(points=keep_pts)
    fw_s    = Fsurf_data["fw_comp"].stack(points=("y", "x")).sel(points=keep_pts)
    rho_s   = Fsurf_data["rho"].stack(points=("y", "x")).sel(points=keep_pts)

    # areacello (pick the first one if your area_ds stores a list per model)
    area1 = area_ds[model]
    area_s = area1.stack(points=("y", "x")).sel(points=keep_pts)

    # -------------------------
    # 3) Apply weights (still xarray), then convert to NumPy once
    #    Keep as (time, points) for speed (no need to unstack!)
    # -------------------------
    wf_s  = (fsurf_s * area_s)
    wh_s  = (heat_s  * area_s)
    wfw_s = (fw_s    * area_s)

    # OPTIONAL: restrict time for testing (matches your i-model testing)
    """wf_s  = wf_s.isel(time=slice(0, 1))
    wh_s  = wh_s.isel(time=slice(0, 1))
    wfw_s = wfw_s.isel(time=slice(0, 1))
    rho_s = rho_s.isel(time=slice(0, 1))"""

    timepoints = rho_s["time"].values

    # Convert to numpy arrays ONCE (prevents pandas/dask recursion issues)
    wf_np   = wf_s.values      # shape (time, points)
    wh_np   = wh_s.values
    wfw_np  = wfw_s.values
    rho_np  = rho_s.values     # density at points
    area_np = area_s.values    # shape (points,)

    # -------------------------
    # 4) Main loops (time x rho_bins), vectorized across points
    # -------------------------
    rows = []
    for t in range(len(timepoints)):
        rho_t = rho_np[t, :]

        for rhoclass in rho_classes:
            rhobot = rhoclass
            rhotop = rhoclass + step_size

            idx = np.where((rho_t > rhobot) & (rho_t < rhotop))[0]

            fgen_value  = float(np.nansum(wf_np[t, idx])  / step_size / 1e6)
            heat_value  = float(np.nansum(wh_np[t, idx])  / step_size / 1e6)
            fw_value    = float(np.nansum(wfw_np[t, idx]) / step_size / 1e6)
            area_sum    = float(np.nansum(area_np[idx]))

            rows.append([t, rhoclass + step_size/2, fgen_value, heat_value, fw_value, area_sum])

    Fgen = pd.DataFrame(rows, columns=["time", "rho", "Fgen", "HeatFlux", "FreshwaterFlux", "AreaSum"])
    Fgen = Fgen.groupby("rho", as_index=False)[["Fgen", "HeatFlux", "FreshwaterFlux", "AreaSum"]].mean()

    Fgen_dict[model] = Fgen
    print(f"Completed Fgen calculation for {model}")


# %% [markdown]
# #### (lat, lon) models

# %%
for model, Fsurf_data in Fsurf_datasets.items():
    if model not in norm_models:
        continue

    filtered_Fsurf = (
        Fsurf_data
        .where(Fsurf_data["lat"] > 45, drop=True)
    )#.isel(time=slice(9, 13))
    

    fsurf = filtered_Fsurf["fsurf"]
    heat_comp = filtered_Fsurf["heat_comp"]
    fw_comp = filtered_Fsurf["fw_comp"]
    rho = filtered_Fsurf["rho"]

    lat_idx = np.where(Fsurf_data["lat"].values > 45)[0]
    area1 = area_ds[model].isel(lat=lat_idx)

    timepoints = filtered_Fsurf["time"].values

    #weighted_fsurf = fsurf * area1
    #weighted_heat_comp = heat_comp * area1
    #weighted_fw_comp = fw_comp * area1

    rho_np, fsurf_np, heat_np, fw_np, area_v = dask.compute(
        rho.data,        # (time, lat, lon)
        fsurf.data,
        heat_comp.data,
        fw_comp.data,
        area1.data,      # (lat, lon)
    )

    # numpy area
    area_v = area1.values

    rows = []
    for time in range(len(timepoints)):
        rho_t = rho_np[time]
        fsurf_t = fsurf_np[time]
        heat_comp_t = heat_np[time]
        fw_comp_t = fw_np[time]
        wf_t  = fsurf_t * area_v
        wh_t  = heat_comp_t * area_v
        wfw_t = fw_comp_t * area_v

        print(time)

        for rhoclass in rho_classes:
            rhobot = rhoclass
            rhotop = rhoclass + step_size

            m = (rho_t > rhobot) & (rho_t < rhotop)   # boolean (lat, lon)
            

            fgen_value = float(np.nansum(wf_t[m])  / step_size / 1e6)
            heat_value = float(np.nansum(wh_t[m])  / step_size / 1e6)
            fw_value   = float(np.nansum(wfw_t[m]) / step_size / 1e6)
            area_sum   = float(np.nansum(area_v[m]))

            rows.append([time, rhoclass + step_size/2, fgen_value, heat_value, fw_value, area_sum])

    Fgen = pd.DataFrame(rows, columns=["time", "rho", "Fgen", "HeatFlux", "FreshwaterFlux", "AreaSum"])
    Fgen = Fgen.groupby("rho", as_index=False)[["Fgen", "HeatFlux", "FreshwaterFlux", "AreaSum"]].mean()

    Fgen_dict[model] = Fgen
    print(f"Completed Fgen calculation for {model}")


# %%
import pickle

save_path = "/glade/work/stevenxu/AMOC_models/Fgen_Allmodels.pkl"
with open(save_path, "wb") as f:
    pickle.dump(Fgen_dict, f)

