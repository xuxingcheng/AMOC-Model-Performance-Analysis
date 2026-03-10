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


# %% [markdown]
# # Data Import

# %%
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Not a valid ID")

excluding_models = ["CESM2", 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'FGOALS-g3', 'CanESM5-1', 'GISS-E2-2-G', 'NorESM2-LM']# models not included in calculation
#excluding_models = []
models = []

def dataconcat(scenario, variable):
    dir_path = f"/glade/work/stevenxu/AMOC_models/{variable}/scenarios/{scenario}"
    all_files = glob.glob(os.path.join(dir_path, "*.nc"))

    groups = defaultdict(list)
    for fp in all_files:
        fname = os.path.basename(fp)
        model_name = fname.split("_")[2]
        groups[model_name].append(fp)

    datasets = {}
    for prefix, files in groups.items():
        if prefix in excluding_models:
            continue

        files = sorted(files)
        print(f"Testing files for {prefix}")

        clean_files = []
        for f in files:
            # skip zero-byte files immediately
            if os.path.getsize(f) == 0:
                print(f"  Skipping empty file for {prefix}: {f}")
                continue
            try:
                xr.open_dataset(f, engine="netcdf4").close()
                clean_files.append(f)
            except Exception as e:
                print(f"  Skipping unreadable file for {prefix}: {f}")
                print("    Error:", repr(e))

        if not clean_files:
            print(f"⚠ No valid files left for {prefix}, skipping model.")
            continue

        print(f"Concatenating {len(clean_files)} valid files for {prefix}")
        ds = xr.open_mfdataset(
            clean_files,
            combine="by_coords",
            parallel=True,
            engine="netcdf4",
            use_cftime=True,
            chunks={"time": 12},   # add this
        )
        datasets[prefix] = ds
        models.append(prefix)

    return datasets
              
sst_datasets = dataconcat("PIControl", "sea_surface_temperature") 
sss_datasets = dataconcat("PIControl", "sea_surface_salinity") 
hf_datasets = dataconcat("PIControl", "heatflux") 
wf_datasets = dataconcat("PIControl", "waterflux")
models = set(models)

# %%
def subtract_years_cftime(t, years):
    """Return a new cftime object with years subtracted (same month/day)."""
    return type(t)(
        t.year - years, t.month, t.day,
        t.hour, t.minute, t.second, t.microsecond,
        has_year_zero=t.has_year_zero
    )

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

    end_times = []
    for variable_ds in [sst_datasets, sss_datasets, hf_datasets, wf_datasets]:
        end_times.append(variable_ds[model]['time'].isel(time=-1).values.item())

    start_time = subtract_years_cftime(min(end_times), lastNyear)
    end_time   = min(end_times)
    print(f"Aligning time for {model} to last {lastNyear} years: {start_time} to {end_time}")

    sst_datasets[model] = sst_datasets[model].sel(time=slice(start_time, end_time))
    sss_datasets[model] = sss_datasets[model].sel(time=slice(start_time, end_time))
    hf_datasets[model]  = hf_datasets[model].sel(time=slice(start_time, end_time))
    wf_datasets[model]  = wf_datasets[model].sel(time=slice(start_time, end_time))


for model in models:
    align_time(model, 20)

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

# %%
ij_models = []
i_models = []
norm_models = []

for model_name, ds in Fsurf_datasets.items():
    dims = set(ds.dims)

    if {'i', 'j'}.issubset(dims):
        ij_models.append(model_name)
    elif 'i' in dims and 'j' not in dims:
        i_models.append(model_name)
    elif {'lat', 'lon'}.issubset(dims):
        norm_models.append(model_name)

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
dir_path = "/glade/work/stevenxu/AMOC_models/areacello"
all_files = glob.glob(os.path.join(dir_path, "*.nc")) 

area_ds = defaultdict(list)
model_names = []
for fp in all_files: 
    fname = os.path.basename(fp) 
    model_name = fname.split("_")[2] 
    area  = xr.open_dataset(fp)["areacello"]
    model_names.append(model_name)
    area_ds[model_name].append(area)
	

# %% [markdown]
# Use ACCESS one for missing areacello

# %%
area_ds['FGOALS-f3-L'] = [da.copy(deep=True) for da in area_ds['ACCESS-CM2']]

# %% [markdown]
# Align areacello and actual data (180,288 -> 90, 144)

# %%
fs = Fsurf_datasets["GISS-E2-1-G-CC"]
area0 = area_ds["GISS-E2-1-G-CC"][0]          # (lat:180, lon:288)

# 1) coarsen to (lat:90, lon:144) — use SUM for area
area_coarse = area0.coarsen(lat=2, lon=2, boundary="trim").sum()

# 2) align to Fsurf grid (safe even if tiny coord diffs)
area_coarse = area_coarse.sel(
    lat=fs["lat"],
    lon=fs["lon"],
    method="nearest"
)

# optional: sanity check
print("Fsurf grid:", fs.sizes["lat"], fs.sizes["lon"])
print("Area  grid:", area_coarse.sizes["lat"], area_coarse.sizes["lon"])

# 3) save it for later use
area_ds["GISS-E2-1-G-CC"][0] = area_coarse

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

    area1 = area_ds[model][0].sel(i=filtered_Fsurf["i"])  # <-- critical alignment

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
    area1 = area_ds[model][0]
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
    area1 = area_ds[model][0].isel(lat=lat_idx)

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


