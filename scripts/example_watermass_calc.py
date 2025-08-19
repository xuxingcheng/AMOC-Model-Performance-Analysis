import numpy as np
from netCDF4 import Dataset
from math import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import sys
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri
import os
from scipy import optimize
from scipy.signal import savgol_filter
import gsw
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# don't worry about this function, it's just for reading data that's stored in an annoying way 
def read_data(var,direct,start_date,return_lat=False):
	
	dir1 = direct
	dir1_datalist = []
	counter = 0
	for filename in enumerate(sorted(os.listdir(dir1))):
		if filename[1][-12:-8] in ['h.24']: #,'h.3','h.4','h.5']:#,'h.5','h.6']:#'h.8','h.9']:
			if (int(filename[1][-10:-6]) > start_date):
				if counter %12 ==0:
					print(filename)
				file = os.path.join(dir1,filename[1])
				ncfile = Dataset(file, 'r')

				lat = ncfile.variables['TLAT'][:]
				lon = ncfile.variables['TLONG'][:]
				mask = ncfile.variables['REGION_MASK'][:]
				area = ncfile.variables['TAREA'][:]/10000 #convert to meters 
				z = ncfile.variables['z_t'][:]/100 #convert to meters
				if var in ['TEMP','SALT','PD']:
					temp_var = ncfile.variables[var][:,0,:,:] #take only surface
				else:
					temp_var = ncfile.variables[var][:,:]
				temp_var = np.where(temp_var>1.0e18,np.nan,temp_var)
			
				dir1_datalist.append(temp_var)
				counter+=1

	var_out = np.concatenate(dir1_datalist,axis=0)
	if return_lat == True:
		return [var_out,lat,lon,z,area,mask]
	else:
		return var_out  

PI_dir = '/glade/campaign/univ/uwas0157/preindustrial_spinup/ocn/hist/tmp.nc_compress/'

# this is a dictionary that contains all the different variables I'll need. I didn't have a 'total freshwater'
# variable so I had to calculate it myself as the sum of many other terms
PI_dict = {}
vars = ['SHF','PREC_F','EVAP_F','QFLUX','MELT_F','IOFF_F','ROFF_F','SALT_F','PD']#
for i,var in enumerate(vars):
    PI_dict[var] = read_data(var,PI_dir,2480,False)
PI_dict['QFLUX'] = -PI_dict['QFLUX']/Lv/1e4
PI_dict['FW'] =PI_dict['PREC_F']+PI_dict['EVAP_F']+PI_dict['ROFF_F']+PI_dict['IOFF_F'] +PI_dict['MELT_F'] + PI_dict['QFLUX']+PI_dict['SALT_F']*sflux_factor/sal_factor


# Read in some constants from a netcdf file-- you might have to just look these constants up
file = os.path.join(direct,'hist',exper+'.pop.h.2501-01.nc')
ncfile = Dataset(file, 'r')
Lv =ncfile.variables['latent_heat_vapor']
sal_factor = ncfile.variables['salinity_factor'][:]
sflux_factor = ncfile.variables['sflux_factor'][:]
cp_sw = ncfile.variables['cp_sw'][:]*1e-4 # go from erg/g/K to J/kg/K
rho_sw = ncfile.variables['rho_sw'][:]*1e3 # go from g/cm^3 to kg/m^3
rho_fw = ncfile.variables['rho_fw'][:]*1e3
S0  = ncfile.variables['ocn_ref_salinity'][:] #in g/kg
mask = ncfile.variables['REGION_MASK'][:] # this tells you what ocean basin the data is in


# looks like I originally calculated these seawater expansion coefficients from the Gibbs Seawater Package using mean 
# surface temp and salt and then saved them manually
alpha = 0.0001391022569339093
beta = 0.0007600059551595953
# alpha = gsw.density.alpha(np.nanmean(S_sfc[:,rgn[0],rgn[1]]),np.nanmean(T_sfc[:,rgn[0],rgn[1]]),0)
# beta = gsw.density.beta(np.nanmean(S_sfc[:,rgn[0],rgn[1]]),np.nanmean(T_sfc[:,rgn[0],rgn[1]]),0)



# This chunk of code sets up the discrete array of potential densities ('PD') that the Fgen will be calculated for
PD_timeavg = np.nanmean(PI_dict['PD'],axis=0)
# here I looked at the time average potential densities over the region of the north atlantic
# and picked the minimum and maximum from that as guidance for my array values. There's probably a better way to do this.
NA_region_PDs = PD_timeavg[np.where((lat>45)&(np.isin(mask,np.array([6.0,8.0,9.0,10.0]))))]
min_PD = np.nanmin(NA_region_PDs)
max_PD = np.nanmax(NA_region_PDs)+.0005 # added this buffer because the monthly max PD can be greater than the max of the timeaverage
PD_step = .00005
PD_range = np.arange(min_PD-PD_step,max_PD+PD_step,PD_step) #added more buffer here to the min

#convert the buoyancy fluxes to the same units
#need to decide on a sign convention. The fluxes themselves are positive = more buoyancy (less sinking)
#so here I add a negative sign when calculating Fgen because we want more buoyancy = less strong AMOC (negative change)
ffw = -rho_sw/rho_fw*beta*S0*PI_dict['FW']
fheat = -alpha/cp_sw*PI_dict['SHF']

# set up arrays with dimensions of time and potential density class
ffw_profile = np.zeros((PI_dict['SHF'].shape[0],len(PD_range)))
fheat_profile = np.zeros((PI_dict['SHF'].shape[0],len(PD_range)))

# loop over months and density classes
for i,PD_class in enumerate(PD_range):
    for month in range(PI_dict['SHF'].shape[0]):
    	# find the grid points where the surface density falls in the given density class
        rgn = np.where((PI_dict['PD'][month]>PD_class)&(PI_dict['PD'][month]<PD_class+step)&(lat>30)&(np.isin(mask,np.array([6.0,8.0,9.0,10.0]))))
        # calculate d/dsigma of the area integral in the Fgen equation. This just comes out to summing the buoyancy flux over the region
        # where the surface densities fall within the given density class, and dividing by the step between density classes 
        ffw_profile[month,i] = np.nansum(np.multiply(ffw[month,rgn[0],rgn[1]],area[rgn[0],rgn[1]]))/PD_step/1e3 #because step is in units of g/cm^3 and we need SI units
        fheat_profile[month,i] = np.nansum(np.multiply(fheat[month,rgn[0],rgn[1]],area[rgn[0],rgn[1]]))/PD_step/1e3


#take the time mean because I just wanted to see what this looks like on average for the preindustrial
ffw_profile_mean = np.nanmean(ffw_profile,axis=0)
fheat_profile_mean = np.nanmean(fheat_profile,axis=0)

plt.figure(1,figsize=(8,11))
plt.plot(ffw_profile_mean/1e6,PD_range+PD_step/2,label='freshwater') #divide by 1e6 to convert to Sv
plt.plot(fheat_profile_mean/1e6,PD_range+PD_step/2,label='heating')
plt.plot((ffw_profile_mean+fheat_profile_mean)/1e6,PD_range+PD_step/2,lw=2,label='total',color='black') # add the two contributions together to get the total Fgen
plt.gca().invert_yaxis()
plt.legend()
plt.axvline(0,linestyle='--',color='gray')
plt.xlim(-26,10)
plt.xlabel('Surface flux above 45N (Sv)')
plt.ylabel('Potential density with reference to the surface (g/cm$^3$)')
plt.title('Average over Amoc min 20 years of gradual meltf ramp')
plt.savefig('../figures/water_mass_trans/profiles/profile_preindustrial.png',bbox_inches='tight')

