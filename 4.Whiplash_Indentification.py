#########################################################################################
## Identify whiplash occurrences for all grid points across the CONUS from 1915 to 2020.
## Bryony Louise Puxley
## Last Edited: Friday, July 25th, 2025 
## Input: Regional SPI files or CONUS-wide SPI file - need all times at each grid point.
## Output: netCDF file of identified whiplash occurrences at a ~ 6 km grid resolution 
## from 1915 to 2020, and a PNG file of the number of whiplash occurrences across the 
## chosen region for visualization.
#########################################################################################
# Import Required Modules
#########################################################################################
import xesmf as xe
import numpy as np
import xarray as xr
from tqdm import tqdm
import time
from datetime import datetime, timedelta, date
from netCDF4 import Dataset, num2date, MFDataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import scipy.stats as scs
import os
import gzip

#########################################################################################
# Import Functions
#########################################################################################
import functions 
#########################################################################################
# Regions For Analysis
#########################################################################################
#Choose Region
Region = "SGP"

region_lon = {"CONUS":[235,295],
              "WCN":[235,241], #west coast north
              "WCS":[235,241], #west coast south
              "MWN":[241,255], #mountain west north
              "MWC":[241,255], #moutain west central
              "MWS":[241,255], #mountain west south
              "NGP":[255,265], #northern great plains
              "SGP":[255,265], #southern great plains
              "NGL":[265,279], #northern great lakes
              "SGL":[265,279], #southern great lakes
              "NNE":[279,295], #northern north east
              "SNE":[279,295], #southern north east
              "WSE":[265,275], #western south east
              "ESE":[275,286]} #eastern south east

region_lat = {"CONUS":[25,50],
              "WCN":[40,50], #west coast north
              "WCS":[25,40], #west coast south
              "MWN":[42,50], #mountain west north 
              "MWC":[33,42], #moutain west central
              "MWS":[25,33], #moutain west south              
              "NGP":[40,50], #northern great plains
              "SGP":[25,40], #southern great plains
              "NGL":[43,50], #northern great lakes
              "SGL":[36,43], #southern great lakes
              "SNE":[36,43], #southern north east
              "NNE":[43,50], #northern north east
              "WSE":[25,36], #western south east 
              "ESE":[25,36]} #eastern south east
              
inputlon = region_lon[Region]
inputlat = region_lat[Region]

#########################################################################################
# Import Data
#########################################################################################
window = 30 #choose from 60-, 90-, or 180
filename = 'SPI_%s_%s.nc'%(window, Region)
dirname= '/scratch/bpuxley/SPI_30day'

pathfile = os.path.join(dirname, filename)

df_spi = xr.open_dataset(pathfile)
print('Read in Data')

m,o,p = len(df_spi.time), len(df_spi.lat), len(df_spi.lon) #length of data #mo,o,p: time, lat, lon 
lon, lat = np.meshgrid(df_spi.lon.values, df_spi.lat.values) #create a meshgrid of lat,lon values

print('Time: '+ str(m))
print('Lat: '+ str(o))
print('Lon: '+ str(p))

#########################################################################################
# Loop through each grid point and identify occurrences of whiplashes
#########################################################################################
spi = df_spi.spi_30_day #create variable spi which contains the spi data from the data array
				
DPcount = np.zeros((o,p)) #create an array that is lat by lon to store the No of Events at each grid point.
PDcount = np.zeros((o,p)) #create an array that is lat by lon to store the No of Events at each grid point.

binary_array_DP = np.zeros((m,o,p))*np.nan #create an array to store the binary array of whiplash occurrences
binary_array_PD = np.zeros((m,o,p))*np.nan #create an array to store the binary array of whiplash occurrences

#Loop through all days and identify the grid points that experience whiplash events 
print('Starting loop to identify whiplash events')
for i in tqdm(range(window-1, m-window)): 
#i.e, for rolling 30-day periods; 
## loop through the timeseries from value 29 (first 30 values will be all nans) up until the last 30 days
#Look for DP events by comparing the SPI value at day i to the SPI value at day i+30
#day i is the 30 day SPI from day i-30 to day i; day i+30 is the 30 day SPI from day i to day i+30
#If a window of 60-,90-, or 180- days was chosen, just replace 30 with that value
	
	#drought-to-pluvial whiplash events
	bool_array = xr.where((spi[i].values <= -1) & (spi[i+window].values >= +1),True,False) #find all the grid points that experience a whiplash event
	binary_array_DP[i,:,:] = bool_array
	
	DPcount = DPcount + np.where(bool_array,1,0) #count the number of whiplash events at each grid point
	
	#pluvial-to-drought whiplash events
	bool_array = xr.where((spi[i].values >= +1) & (spi[i+window].values <= -1),True,False) #find all the grid points that experience a whiplash event
	binary_array_PD[i,:,:] = bool_array
	
	PDcount = PDcount + np.where(bool_array,1,0) #count the number of whiplash events at each grid point

print('whiplash events identifited')
#########################################################################################
# Save out the binary file of grid points meeting whiplash criteria
#########################################################################################
time = pd.date_range(start='1915-01-01', end='2020-12-31', freq='D')
lats = df_spi.lat
lons = df_spi.lon
attrs = df_spi.attrs

dimensions=['time', 'lat', 'lon']
coords = {
			'time': time,
			'lat': lats,
			'lon': lons
			}

DP_xarray = xr.DataArray(binary_array_DP, coords, dims=dimensions, name='Drought to Pluvial')
PD_xarray = xr.DataArray(binary_array_PD, coords, dims=dimensions, name='Pluvial to Drought')
DP_count = xr.DataArray(DPcount, coords={'lat':lats, 'lon':lons}, dims=['lat','lon'], name='Drought to Pluvial Count')
PD_count = xr.DataArray(PDcount, coords={'lat':lats, 'lon':lons}, dims=['lat','lon'], name='Pluvial to Drought Count')

whiplash_dataset = xr.Dataset({"DP_whiplashes":DP_xarray, "PD_whiplashes":PD_xarray, "DP_count":DP_count, "PD_count":PD_count})

whiplash_dataset.to_netcdf('/scratch/bpuxley/Whiplash/whiplashes_%s.nc'%(Region))
print('whiplash file saved')

#########################################################################################
#Plot the count of whiplash events across the region
#########################################################################################
fig = plt.figure(figsize = (6,7), dpi = 300, tight_layout =True)

# First subplot with the count of drought-to-pluvial whiplash number
ax1 = fig.add_subplot(211, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs = plt.contourf(lon, lat, DPcount, transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 900, 2), cmap = 'hot_r') 
fig.colorbar(cs, ax=ax1, orientation='horizontal', pad=0.05)

plt.title('Drought-to-Pluvial')

# Second subplot with the count of pluvial-to-drought whiplash number
ax2 = fig.add_subplot(212, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linewidth=1)
ax2.add_feature(cfeature.STATES, edgecolor='black')

ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs = plt.contourf(lon, lat, PDcount, transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 900, 2), cmap = 'hot_r') 
fig.colorbar(cs, ax=ax2, orientation='horizontal', pad=0.05)

plt.title('Pluvial-to-Drought')

plt.savefig('/scratch/bpuxley/Plots/Whiplashes_%s_1915_2020'%(Region), bbox_inches = 'tight', pad_inches = 0.1)    
