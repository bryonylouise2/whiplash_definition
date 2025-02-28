#########################################################################################
## Examine and Plot Specific Events
## Bryony Louise
## Last Edited: Wednesday November 13th 2024 
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
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import spei as si
import pandas as pd
import scipy.stats as scs
import os

#########################################################################################
#Dates for Examination
#########################################################################################
#Drought-to-Pluvial
DP_date_drought = "1938-12-18"
DP_date_pluvial = str(date(1938, 12, 18) + timedelta(days=30))

#Pluvial-to-Drought
PD_date_pluvial = "1991-09-24"
PD_date_drought = str(date(1991, 9, 24) + timedelta(days=30))

#########################################################################################
#Import Data
#########################################################################################
Regions = {"WCN", "WCS", "MWN", "MWC", "MWS", "NGP", "SGP", "NGL", "SGL", "NNE", "SNE", "ESE", "WSE"}
dirname= '/scratch/bpuxley/SPI_30day'

spi_pathfiles = []
for i in Regions:
        filename = 'SPI30_%s.nc'%(i)
        pathfile = os.path.join(dirname, filename)
        spi_pathfiles.append(pathfile)

whiplash_pathfiles = ['/scratch/bpuxley/Whiplash/whiplashes_CONUS_1915_1967.nc', '/scratch/bpuxley/Whiplash/whiplashes_CONUS_1968_2020.nc']
density_pathfiles = ['/scratch/bpuxley/Density/density_1935_1944.nc', '/scratch/bpuxley/Density/density_1988_1997.nc']

df_spi = xr.open_mfdataset(spi_pathfiles, combine='by_coords')
df_whiplash = xr.open_mfdataset(whiplash_pathfiles, combine='by_coords')
df_density_DP = xr.open_dataset(density_pathfiles[0])
df_density_PD = xr.open_dataset(density_pathfiles[1])

print('Read in Data')

#########################################################################################
#Select only dates required for examination
#########################################################################################
#Drought-to-Pluvial
df_spi_dp_drought = df_spi.spi_30day.sel(time=DP_date_drought)
df_spi_dp_pluvial = df_spi.spi_30day.sel(time=DP_date_pluvial)
df_whiplash_dp = df_whiplash.DP_whiplashes.sel(time=DP_date_drought)
df_density_dp = df_density_DP.DP_density.sel(time=DP_date_drought)

#Pluvial-to-Drought
df_spi_pd_pluvial = df_spi.spi_30day.sel(time=PD_date_pluvial)
df_spi_pd_drought = df_spi.spi_30day.sel(time=PD_date_drought)
df_whiplash_pd = df_whiplash.PD_whiplashes.sel(time=PD_date_pluvial)
df_density_pd = df_density_PD.PD_density.sel(time=PD_date_pluvial)

#########################################################################################
#Filter date for plotting
#########################################################################################
#Drought areas where SPI values are less than negative 1
spi_dp_drought = xr.where(df_spi_dp_drought <= -1,True,False)
spi_pd_drought = xr.where(df_spi_pd_drought <= -1,True,False)

#Pluvial areas where SPI values are greater than positive 1
spi_dp_pluvial = xr.where(df_spi_dp_pluvial >= 1,True,False)
spi_pd_pluvial = xr.where(df_spi_pd_pluvial >= 1,True,False)

#########################################################################################
#Plot the Event
#########################################################################################
lon, lat = np.meshgrid(df_spi.lon, df_spi.lat)
#########################################################################################
#Drought-to-Pluvial
#########################################################################################
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

# First subplot of the area with SPI <-1 (Drought)
ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

# Create a custom color map: brown for True, white for False
cmap = ListedColormap(["white", "brown"])

cs1 = ax1.pcolormesh(lon[:,:], lat[:,:], spi_dp_drought[:,:], cmap=cmap, transform=ccrs.PlateCarree())

plt.title('Areas where SPI <-1: Nov 18 and Dec 18, 1938')

# Second subplotof the area with SPI >+1 (Pluvial)
ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linewidth=1)
ax2.add_feature(cfeature.STATES, edgecolor='black')

ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

# Create a custom color map: blue for True, white for False
cmap = ListedColormap(["white", "blue"])

cs2 = ax2.pcolormesh(lon[:,:], lat[:,:], spi_dp_pluvial[:,:], cmap=cmap, transform=ccrs.PlateCarree())

plt.title('Areas where SPI <-1: Dec 18, 1938 and Jan 17, 1939')

# Third subplot with the whiplash locations
ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax3.add_feature(cfeature.COASTLINE)
ax3.add_feature(cfeature.BORDERS, linewidth=1)
ax3.add_feature(cfeature.STATES, edgecolor='black')

ax3.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

# Create a custom color map: purple for True, white for False
cmap = ListedColormap(["white", "purple"])

cs3 = ax3.pcolormesh(lon[:,:], lat[:,:], df_whiplash_dp[:,:], cmap=cmap, transform=ccrs.PlateCarree())

plt.title('Whiplash Occurrences')

# Fourth subplot with the normalized density and polygon
ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax4.add_feature(cfeature.COASTLINE)
ax4.add_feature(cfeature.BORDERS, linewidth=1)
ax4.add_feature(cfeature.STATES, edgecolor='black')

ax4.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs4 = plt.contourf(lon[:,:], lat[:,:], df_density_dp[:,:], transform=ccrs.PlateCarree(), levels=np.arange(0, 1.1, 0.1), cmap = 'Purples') 
fig.colorbar(cs4, ax=ax4, orientation='horizontal', pad=0.05)

plt.title('Spatial Density')

plt.show(block=False)

plt.savefig('/scratch/bpuxley/Plots/event_1938_12_18', bbox_inches = 'tight', pad_inches = 0.1)    
