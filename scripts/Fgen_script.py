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


def dataconcat (model, scenario, variable): # path of the folder containing the netCDF files 
    dir_path = f"/glade/work/stevenxu/AMOC_models/FgenCalculation/{model}/{variable}/{scenario}"
    all_files = glob.glob(os.path.join(dir_path, "*.nc")) 
    
    groups = defaultdict(list) 
    for fp in all_files: 
        fname = os.path.basename(fp) 
        model_name = fname.split("_")[2] 
        groups[model_name].append(fp) 
        datasets = {}

    for prefix, files in groups.items(): 
        files = sorted(files) 
        print(f"Concatenating {len(files)} files for {prefix}") 
        ds = xr.open_mfdataset( files, combine="by_coords", parallel=True ) 
        datasets[prefix] = ds.isel(time=slice(-20 * 12, None)) 
    
    return datasets 


# -----------------------------------------------------------------------------------------------
# Sea surface density
def compute_surface_density(model, sst_datasets, sss_datasets, last_n_months=None):
    T  = sst_datasets[model]['tos']          
    SP = sss_datasets[model]['sos']          
    lon = sst_datasets[model]['longitude']
    lat = sst_datasets[model]['latitude']
    # addubg vertices for accurate area-weighted calculations for next step
    vertices_latitude = sst_datasets[model]['vertices_latitude']
    vertices_longitude = sst_datasets[model]['vertices_longitude']

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



    return xr.Dataset(dict(rho=rho, alpha=alpha, beta=beta, vertices_latitude=vertices_latitude, vertices_longitude=vertices_longitude))

# -----------------------------------------------------------------------------------------------
# Fsurf calculation

def compute_fsurf(model,
                  sst_datasets, sss_datasets, hf_datasets, wf_datasets,
                  cp=3990.0, rho0=1027.0, rho_fw=1000.0, S0=35.0,
                  last_n_months=None):

    HF = hf_datasets[model]['hfds']  # W m^-2, 
    WF = wf_datasets[model]['wfo']     # kg m^-2 s^-1, 
    lon = sst_datasets[model]['longitude']
    lat = sst_datasets[model]['latitude']
    p0 = 0.0
    # addubg vertices for accurate area-weighted calculations for next step
    vertices_latitude = sst_datasets[model]['vertices_latitude']
    vertices_longitude = sst_datasets[model]['vertices_longitude']

    density_data = compute_surface_density(model, sst_datasets, sss_datasets, last_n_months=last_n_months)
    rho = density_data['rho']
    alpha = density_data['alpha']
    beta = density_data['beta']

    if last_n_months is not None:
        HF  = HF.isel(time=slice(-last_n_months, None))
        WF = WF.isel(time=slice(-last_n_months, None))

    # f_surf = -(alpha/cp) * f_heat  - (rho0/rho_fw) * beta * S0 * f_water
    fsurf = -(alpha / cp) * HF  -  (rho0 / rho_fw) * beta * S0 * WF
    fsurf = fsurf.assign_attrs(
        long_name="Buoyancy-relevant surface forcing (Eq. 5)",
        description="-(alpha/cp)*f_heat - (rho0/rho_fw)*beta*S0*f_water",
        units="",
        cp=cp, rho0=rho0, rho_fw=rho_fw, S0=S0
    )

    heat_comp = -(alpha / cp) * HF
    fw_comp = -(rho0 / rho_fw) * beta * S0 * WF

    return xr.Dataset(dict(fsurf=fsurf, rho=rho, heat_comp=heat_comp, fw_comp=fw_comp, vertices_latitude=vertices_latitude, vertices_longitude=vertices_longitude))


# -----------------------------------------------------------------------------------------------  
# Task:

sst_datasets = dataconcat("ACCESS-CM2", "PIControl", "sea_surface_temperature") 
sss_datasets = dataconcat("ACCESS-CM2", "PIControl", "sea_surface_salinity") 
hf_datasets = dataconcat("ACCESS-CM2", "PIControl", "heatflux") 
wf_datasets = dataconcat("ACCESS-CM2", "PIControl", "waterflux")

# calculate surface density and fsurf
surf_den_ACCESS = compute_surface_density("ACCESS-CM2", sst_datasets, sss_datasets, last_n_months=240)
out_ACCESS = compute_fsurf(
    "ACCESS-CM2",
    sst_datasets=sst_datasets,
    sss_datasets=sss_datasets,
    hf_datasets=hf_datasets,
    wf_datasets=wf_datasets,
    last_n_months=240
)

# Fgen calculation
    # density intervals
rho_min = float(out_ACCESS['rho'].isel(time=0).min())
rho_max = float(out_ACCESS['rho'].isel(time=0).max())
step_size = 0.1
rho_classes = np.arange(rho_min, rho_max + step_size, step_size)

    # area with j,i coordinate data
area  = xr.open_dataset("/glade/work/stevenxu/AMOC_models/areacello/ACCESS-CM2/areacello_Ofx_ACCESS-CM2_piControl_r1i1p1f1_gn.nc")["areacello"]

    # create a mask to filter by latitude
lat   = out_ACCESS["latitude"].stack(points=("j","i"))
mask_pts = lat.where(lat > 30, drop=True).points

    # stack or flatten data into 1d array
fsurf = out_ACCESS["fsurf"].stack(points=("j","i")).sel(points=mask_pts)
rho   = out_ACCESS["rho"].stack(points=("j","i")).sel(points=mask_pts)
area  = area.stack(points=("j","i")).sel(points=mask_pts)

    # weighted
weighted_fsurf = fsurf * area

    # unstack back to original shape
weighted_fsurf = weighted_fsurf.unstack("points")  
rho = rho.unstack("points") 

    # Summing up by dentisy interval
Fgen_org = weighted_fsurf.groupby_bins(rho, bins=rho_classes, right=False).sum(dim=("j","i")) / step_size /1e6

    # adding centered coordinate for density intervals
rho_centers = (rho_classes[:-1] + rho_classes[1:]) / 2

    # organizing
Fgen = Fgen_org.assign_coords(rho_center=("rho_bins", rho_centers))
Fgen = Fgen.rename(rho_bins="rho_intervals")
Fgen = Fgen.rename('Fgen')

    # Saving Fgen data
save_path = "/glade/work/stevenxu/AMOC_models/ACCESS-CM2_Fgen.pkl"
with open(save_path, "wb") as f:
    pickle.dump(Fgen, f)

