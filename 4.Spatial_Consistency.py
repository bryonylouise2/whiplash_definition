#########################################################################################
## Consider the spatial continuity of grid points experiencing a whiplash event to determine events
## Bryony Louise
## Last Edited: Wednesday October 16th 2024 
#########################################################################################
#Import Required Modules
#########################################################################################
import xesmf as xe
import numpy as np
import xarray as xr
import dask
from tqdm import tqdm
import time
from datetime import datetime, timedelta, date
from netCDF4 import Dataset, num2date, MFDataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import spei as si
import pandas as pd
import scipy.stats as scs
import os
from sklearn.neighbors import KernelDensity

from matplotlib.colors import ListedColormap

#########################################################################################
#Import Functions
#########################################################################################
import functions

#########################################################################################
## WHY THESE - ASK TY
#########################################################################################
grid_lons = np.arange(227, 303.01, 0.1)
grid_lats = np.arange(17, 58.01, 0.1)
#########################################################################################

#########################################################################################
#Import Data
#########################################################################################
filename = 'whiplashes.nc'
dirname= '/data2/bpuxley/Whiplash'

pathfile = os.path.join(dirname, filename)

df_whiplash = xr.open_dataset(pathfile)
print('Read in Data')

m,o,p = len(df_whiplash.time), len(df_whiplash.lat), len(df_whiplash.lon) #length of data #mo,o,p: time, lat, lon 
lon, lat = np.meshgrid(df_whiplash.lon.values, df_whiplash.lat.values) #create a meshgrid of lat,lon values

print('Time: '+ str(m))
print('Lat: '+ str(o))
print('Lon: '+ str(p))

#########################################################################################
#Determine Spatial Consistency
#########################################################################################
density_DP = np.zeros((m,o,p))
density_PD = np.zeros((m,o,p))

for i in tqdm(range(29, m-30)): #loop through the timeseries from value 29 (first 30 values will be all nans) up until the last 30 days
	DP_field = df_whiplash.DP_whiplashes[i,:,:].values
	if np.any(DP_field): #only run KDE code if at least one of the values is 1
		density_DP[i, :, :] = functions.kde(orig_lon=df_whiplash.lon.values,
										orig_lat=df_whiplash.lat.values,
               							grid_lon=df_whiplash.lon.values,
                						grid_lat=df_whiplash.lat.values,
                						extreme=DP_field,
                						bandwidth=0.025,
           								)

#########################################################################################
#Save out the density file 
#########################################################################################
time = pd.date_range(start='1915-01-01', end='2020-12-31', freq='D')
lats = df_whiplash.lat
lons = df_whiplash.lon
attrs = df_whiplash.attrs

dimensions=['time','lat','lon']
coords = {
			'time': time,
			'lat': lats,
			'lon': lons
			}
#attributes = {'sources': 'Livneh et al., 2013 & PRISM, 2004', 'references': 'http://www.esrl.noaa.gov/psd/data/gridded/data.livneh.html','}

DP_xarray = xr.DataArray(density_DP, coords, dims=dimensions, name='Drought to Pluvial')
PD_xarray = xr.DataArray(density_PD, coords, dims=dimensions, name='Pluvial to Drought')
density_dataset = xr.Dataset({"DP_density":DP_xarray, "PD_density":PD_xarray})

density_dataset.to_netcdf('/home/bpuxley/Whiplash/density.nc')
print('density file saved')

#########################################################################################
#Plot different days to view events
#########################################################################################
fig = plt.figure(figsize = (6,7), dpi = 300, tight_layout =True)

cmap = ListedColormap(['white', 'purple', 'white'])

# First subplot with the count of drought-to-pluvial whiplash number
ax1 = fig.add_subplot(211, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs = plt.contourf(lon, lat, df_whiplash.DP_whiplashes[428,:,:].values, transform=ccrs.PlateCarree(), levels=np.arange(0, 1.1, 0.1), cmap = 'Purples') 

plt.title('Drought-to-Pluvial')

# Second subplot with thr count of pluvial-to-drought whiplash number
ax2 = fig.add_subplot(212, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linewidth=1)
ax2.add_feature(cfeature.STATES, edgecolor='black')

ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs = plt.contourf(lon, lat, density_DP[428,:,:], transform=ccrs.PlateCarree(), levels=np.arange(0, 1.1, 0.1), cmap = 'Purples') 
fig.colorbar(cs, ax=ax2, orientation='horizontal', pad=0.05)

plt.title('Pluvial-to-Drought')

plt.show(block=False)

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/Whiplashes_%s_1915_2020'%(Region), bbox_inches = 'tight', pad_inches = 0.1)    

