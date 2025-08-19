import numpy as np
from netCDF4 import Dataset
from math import *
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import sys
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri
import matplotlib as mpl
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

aice_file = '/glade/work/camilleh/AMOC-replication-code/raw_data/aice/four_percent_2x_aice.nc'
aice_data = xr.open_dataset(aice_file)
ocn_file = '/glade/campaign/univ/uwas0157/preindustrial_spinup/ocn/hist/tmp.nc_compress/restart_spunup_v15.pop.h.2500-01.nc'
lat = xr.open_dataset(ocn_file)['TLAT'].values
lon = xr.open_dataset(ocn_file)['TLONG'].values

lensim = len(aice_data['time'])
newtime = xr.cftime_range(start='2501-01', periods=lensim, freq='MS', calendar='noleap')
aice_arr = aice_data['aice'].assign_coords(time=newtime)
aice_yearly = aice_arr.groupby('time.year').mean('time')

fig = plt.figure(1,figsize=(8,8))
plt.rcParams.update({'font.size':14})
shrink =.4

# there are many ways to set up the figure: fig.add_subplot is just one option. 
# The important thing is to use something that allows the projection argument.
ax = fig.add_subplot(1,1,1, projection=ccrs.NorthPolarStereo())
# set the longitude and latitude extent you want the map to extend
ax.set_extent([0.0001, 360, 40, 90], crs=ccrs.PlateCarree())
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND)
ax.gridlines(color="black", linestyle="dotted")
ax.set_boundary(circle, transform=ax.transAxes)

# The following chunk of code performs the interpolation onto a regular lat-lon grid. 
x = np.arange(np.nanmin(lon),np.nanmax(lon),3) # new longitude grid you'll use (pick a step size close to the original grid spacing, for you maybe use 1 degree)
y = np.arange(-87,88,3) # new latitude grid you'll project onto 
X,Y = np.meshgrid(x,y) # make the grid 2D 

triang = tri.Triangulation(lon.flatten(),lat.flatten()) # flatten the original 2D irrgular grids into 1D arrays
interpolator =tri.LinearTriInterpolator(triang,aice_yearly.isel(year=slice(40,50)).mean('year').values.flatten()) #set up the interpolation using the tri package
interp_ice = interpolator(X,Y) #perform the interpolation. Now "interp_ice" contains the data variable interpolated onto the new grid.

# The following chunk of code is a somewhat crude way of dealing with the fact that the contour plot
# will show an empty slice of the map between 360 and 0 degrees longitude (it doens't know to wrap the contour/
# data around the zero line). This method for dealing with that is to essentially re-peat a column of data by adding
# the data from the minimum longitude point (let's say it's at 1 degree) to the end of the data (at 361 degrees) so
# that the data wraps around the zero line. 
minLON = x[0]
LON1=X[:,0]*0+360+minLON; 
LON1.shape=(LON1.shape[0],1)
LON2=np.hstack((X,LON1))
LAT1=Y[:,0]; 
LAT1.shape=(LAT1.shape[0],1)
LAT2=np.hstack((Y,LAT1))
ice1=interp_ice[:,1]
ice1.shape=(ice1.shape[0],1)
interp_ice_v2= np.hstack((interp_ice[:,:],ice1))

# plot using the contouf function
out = ax.contourf(LON2,LAT2,interp_ice_v2, transform=ccrs.PlateCarree(),cmap=get_cmap("Blues_r"),levels=np.arange(0,101,5))
fig.colorbar(out,shrink = shrink)
plt.title('Years 40--50')
