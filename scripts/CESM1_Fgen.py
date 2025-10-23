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
import pickle

# %% [markdown]
# # Data Import

# %%
path = '/glade/work/stevenxu/CESM1'
all_files = glob.glob(os.path.join(path, "*.nc"))
print(all_files) 
datasets = {}
for file in all_files:
    varname = os.path.basename(file).split("_")[0]
    datasets[varname] = xr.open_dataset(file)

# %%
PREC = datasets["PREC"]
ROFF = datasets["ROFF"]
SHF  = datasets["SHF"]
MELT = datasets["MELT"]
EVAP = datasets["EVAP"]
IOFF = datasets["IOFF"]
PD   = datasets["PD"]

# %%
prec = PREC['PREC_F']
roff = ROFF['ROFF_F']
melt = MELT['MELT_F']
evap = EVAP['EVAP_F']
ioff = IOFF['IOFF_F']

# Calculate freshwater flux
FWF = (roff + ioff + melt + prec + evap).rename('FWF')
FWF = FWF.assign_attrs({
    'long_name': 'Total Freshwater Flux',
    'units': 'kg/m^2/s',
    'components': 'ROFF_F + IOFF_F + MELT_F + PREC_F + EVAP_F'
})
FWF = ROFF.assign(FWF=FWF)
FWF = FWF.drop_vars('ROFF_F')

# convert density unit from g/cm^3 to kg/m^3
PD["PD"] = PD["PD"] * 1000.0
PD["PD"].attrs["units"] = "kg/m^3"


# %%
SHF

# %%
density_data = PD
hf_datasets = SHF
wf_datasets = FWF

# %%
wf_datasets

# %% [markdown]
# # Calculate F surf

# %% [markdown]
# - Overall mean alpha: 0.00019915005771352584
# - Overall mean beta: 0.000746301819876489

# %%
def compute_fsurf(
				density_data, hf_datasets, wf_datasets,
				cp=3990.0, rho0=1027.0, rho_fw=1000.0, S0=35.0,
				last_n_months=None):

	HF = hf_datasets['SHF']  # W m^-2, 
	WF = wf_datasets['FWF']     # kg m^-2 s^-1, 
	area = wf_datasets["TAREA"]

	rho = density_data['PD']
	alpha = 0.0001391022569339093
	beta = 0.0007600059551595953

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

	return xr.Dataset(dict(fsurf=fsurf, rho=rho, heat_comp=heat_comp, fw_comp=fw_comp, area=area))

Fsurf_data = compute_fsurf(
	density_data = density_data,
	hf_datasets = hf_datasets,
	wf_datasets = wf_datasets,
	last_n_months=240
)

# Drop the z_t coordinate 
Fsurf_data = Fsurf_data.isel(z_t=0).drop_vars('z_t')


Fsurf_data

# %% [markdown]
# # Calculating Fgen at single timepoint

# %% [markdown]
# ### Create density class list
# Take max and min in rho values, and slice with interval 0.01

# %%
ds = Fsurf_data
da = ds['rho'].isel(time=0)
rho_min = float(da.min())
rho_max = float(da.max())

# %%
step_size = 0.05
rho_classes = np.arange(rho_min, rho_max + step_size, step_size)

# %% [markdown]
# ### Dataset for area at each grid cell

# %%
area = Fsurf_data["area"]/1e4
area 

# %% [markdown]
# ### Integration

# %% [markdown]
# Group by density intervals and adding up the area-weighted fsurf

# %%
# create a mask to filter by latitude
lat   = Fsurf_data["TLAT"].stack(points=("nlat","nlon"))
mask_pts = lat.where(lat > 45, drop=True).points

# stack or flatten data into 1d array
fsurf = Fsurf_data["fsurf"].stack(points=("nlat","nlon")).sel(points=mask_pts)
rho   = Fsurf_data["rho"].stack(points=("nlat","nlon")).sel(points=mask_pts)
area  = area.stack(points=("nlat","nlon")).sel(points=mask_pts)

# weighted
weighted_fsurf = fsurf * area

# unstack back to original shape
weighted_fsurf = weighted_fsurf.unstack("points")  
rho = rho.unstack("points") 

# %%
weighted_fsurf

# %%
# Summing up by dentisy interval
Fgen_org = weighted_fsurf.groupby_bins(rho, bins=rho_classes, right=False).sum(dim=("nlat","nlon")) / step_size /1e6

# adding centered coordinate for density intervals
rho_centers = (rho_classes[:-1] + rho_classes[1:]) / 2

# organizing
Fgen = Fgen_org.assign_coords(rho_center=("rho_bins", rho_centers))
Fgen = Fgen.rename(rho_bins="rho_intervals")
Fgen = Fgen.rename('Fgen')
Fgen

# %%
save_path = f"/glade/work/stevenxu/AMOC_models/CESM1_Fgen.pkl"
with open(save_path, "wb") as f:
    pickle.dump(Fgen, f)

# %%
Fgen.mean(dim ='time').isel(rho_intervals = slice(350,600)).plot(x = 'rho_center')

# %%



