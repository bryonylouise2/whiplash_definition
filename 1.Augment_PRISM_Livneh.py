#########################################################################################
## Augment PRISM and Livneh Datasets  
## Bryony Louise
## Last Edited: Wednesday May 29th 2024 
#########################################################################################
#Required Modules for this Function
#########################################################################################
import xesmf as xe
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime, timedelta, date
from netCDF4 import Dataset, num2date, MFDataset

##########################################################################################
#Bilinearly interpolate PRISM precipitation data from a native 4-km grid onto Livneh's grid 
#of about 6 km using Python library xESMF (Zhuang et al. 2020). 
##########################################################################################
def interp(data):
	'''Bilinearly interpolate PRISM grid to Livneh grid.
	
	Parameters
	----------
	data : array
		Array to be interpolated (PRISM)
		
	split : boolean
		If False (default). the time period does not begin in 2011 and end in 2012 (crossover from Livneh to PRISM)

	firstPiece : NoneType or array
        Ignore if split is False. If split is True, an array should be given holding Livneh data.

	'''
	
	prism_pathfile = "/data/deluge/reanalysis/REANALYSIS/PRISM/daily/prec.2020.nc" #gridIn
	livneh_pathfile = "/data/deluge/reanalysis/REANALYSIS/Livneh/prec.2011.nc"  #gridOut

	with Dataset(prism_pathfile, 'r') as nc:
		prismLat = nc.variables['lat'][:]
		prismLon = nc.variables['lon'][:]
		
	with Dataset(livneh_pathfile, 'r') as nc:
		livnehLat = nc.variables['lat'][:]
		livnehLon = nc.variables['lon'][:]
		
	gridIn = {'lat':prismLat, 'lon':prismLon+360}
	gridOut = {'lat':livnehLat, 'lon':livnehLon}
	regridder = xe.Regridder(gridIn, gridOut, method='bilinear')
	prismRegrid = regridder(data)

	return prismRegrid

#########################################################################################
#Import Data
#########################################################################################
#Livneh
#########################################################################################
#open and combine Livneh datasets between 1915 and 2011 into one dataset
year1 = 1915 #first Livneh file wanted
year2 = 2011 #last Livneh file wanted
years = np.arange(year1,year2+1,1) #array of years
livneh_pathfile = [] #list of pathfiles to open
livneh_path = '/data/deluge/reanalysis/REANALYSIS/Livneh/prec.YYYY.nc'

for i in years: #loop through each year
	livneh_pathfile.append(livneh_path.replace("YYYY", str(i)))

df_livneh = xr.open_mfdataset(livneh_pathfile, combine='by_coords') 
#########################################################################################
#PRISM
#########################################################################################
#open and combine PRISM datasets between 2012 and 2020 into one dataset
year1 = 2012 #first PRISM file wanted
year2 = 2020 #last PRISM file wanted
years = np.arange(year1,year2+1,1) #array of years
prism_pathfile = [] #list of pathfiles to open
prism_path = '/data/deluge/reanalysis/REANALYSIS/PRISM/daily/prec.YYYY.nc'

for i in years: #loop through each year
	prism_pathfile.append(prism_path.replace("YYYY", str(i)))

df_prism = xr.open_mfdataset(prism_pathfile, combine='by_coords') 
#########################################################################################
#bilenearly interpolate PRISM precipitation data onto Livneh's grid
#########################################################################################
df_prism_regrid = interp(df_prism)

#########################################################################################
#Save out the PRISM data on its own as a netCDF file
#########################################################################################
df_prism_regrid.to_netcdf('prec.2012_2020_regrid.nc')

#########################################################################################
#Combine and save out the PRISM & livneh datasets as a netCDF file
#########################################################################################
#slice the data to region of interest
livneh_obs = df_livneh.sel(lat=slice(25.16, 49.94)) 
prism_obs = df_prism_regrid.sel(lat=slice(25.16, 49.94))

#Concatenate Livneh and PRISM arrays into one array
prec_ob = xr.concat([livneh_obs, prism_obs], 'time')
time = pd.date_range(start='2010-01-01', end='2020-12-31', freq='D')
lat = prec_ob.lat
lon = prec_ob.lon
shape = (len(time), len(lat), len(lon))

prec_ob.to_netcdf('prec.1915_2020.nc')
