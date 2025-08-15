#########################################################################################
## A script to check the precipitation data to make sure it makes sense.
## Bryony Louise Puxley
## Last Edited: Friday, August 15th, 2025
## Input:
## Output: 
#########################################################################################
#Import Required Modules
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

#########################################################################################
#Import Data
#########################################################################################
filename = 'prec.1915_2020.nc'
dirname= '/home/bpuxley'

pathfile = os.path.join(dirname, filename)

df_precip = xr.open_dataset(pathfile)
print('Read in Data')

#########################################################################################
#Regions For Analysis
#########################################################################################
#Choose Region
Region = "CONUS"

region_lon = {"test":[277,277.2],
              "CONUS":[235,295],
              "WC":[235,241], #west coast
              "MW_N":[241,255], #mountain west north
              "MW_C":[241,255], #moutain west central
              "MW_S":[241,255], #mountain west south
              "NGP":[255,265], #northern great plains
              "SGP":[255,265], #southern great plains
              "NGL":[265,279], #northern great lakes
              "SGL":[265,279], #southern great lakes
              "NNE":[279,295], #northern north east
              "SNE":[279,295], #southern north east
              "WSE":[265,275], #western south east
              "ESE":[275,286]} #eastern south east

region_lat = {"test":[35.1,35.3],
              "CONUS":[25,50],
              "WC":[30,50], #west coast
              "MW_N":[42,50], #mountain west north 
              "MW_C":[33,42], #moutain west central
              "MW_S":[25,33], #moutain west south              
              "NGP":[40,50], #northern great plains
              "SGP":[25,40], #southern great plains
              "NGL":[43,50], #northern great lakes
              "SGL":[36,43], #southern great lakes
              "NNE":[36,43], #northern north east
              "SNE":[43,50], #southern north east
              "WSE":[25,36], #western south east 
              "ESE":[25,36]} #eastern south east

inputlon = region_lon[Region]
inputlat = region_lat[Region]

#slice the data to region of interest
prec_obs = df_precip.sel(lat=slice(inputlat[0], inputlat[1]), lon=slice(inputlon[0], inputlon[1]))
m,o,p = len(prec_obs.time), len(prec_obs.lat), len(prec_obs.lon)
lon, lat = np.meshgrid(prec_obs.lon.values, prec_obs.lat.values) #create a meshgrid of lat,lon values
del(df_precip)

print('Time: '+ str(m))
print('Lat: '+ str(o))
print('Lon: '+ str(p))

#########################################################################################
#Calculate the average precip over the time period at each grid point
#########################################################################################
precip = prec_obs.prec

precip = precip[:,:,:]

m,o,p = precip.shape

precip_sum = np.nansum(precip,axis=0)

precip_avg = precip_sum/105

#########################################################################################
#Plot the Precip data
#########################################################################################
#Calculate the average annual precipitation across the region
fig = plt.figure(figsize = (6,7), dpi = 300, tight_layout =True)

ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs = plt.contourf(lon, lat, precip_avg/25.4, transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 70, 2), cmap = 'terrain_r')
fig.colorbar(cs, ax=ax1, orientation='horizontal', pad=0.05)

plt.title('Average Precip %s 1915-2020'%(Region))

plt.savefig('/scratch/bpuxley/Plots/precip_%s'%(Region), bbox_inches = 'tight', pad_inches = 0.1)


#########################################################################################
