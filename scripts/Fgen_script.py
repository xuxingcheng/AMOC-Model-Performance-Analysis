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

def dataconcat (dir_path): # path of the folder containing the netCDF files 
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
              
sst_datasets = dataconcat("/glade/work/stevenxu/AMOC_models/sea_surface_temperature/scenarios/PIControl") 
sss_datasets = dataconcat("/glade/work/stevenxu/AMOC_models/sea_surface_salinity/scenarios/PIControl") 
hf_datasets = dataconcat("/glade/work/stevenxu/AMOC_models/heatflux/scenarios/PIControl") 
wf_datasets = dataconcat("/glade/work/stevenxu/AMOC_models/waterflux/scenarios/PIControl")

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

surf_den_ACCESS = compute_surface_density("ACCESS-CM2", sst_datasets, sss_datasets, last_n_months=240)

