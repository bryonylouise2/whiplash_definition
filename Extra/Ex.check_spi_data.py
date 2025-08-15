#########################################################################################
## Check the SPI Data to make sure it makes sense
## Bryony Louise Puxley
## Last Edited: Friday, August 15, 2025 
## Input: Regional SPI files.
## Output: A PNG of the annual average SPI across the CONUS.
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
#Regions For Analysis
#########################################################################################
#Choose Region
Region = "CONUS"

Regions = {"WCN", "WCS", "MWN", "MWC", "MWS", "NGP", "SGP", "NGL", "SGL", "NNE", "SNE", "ESE", "WSE"}

region_lon = {"test":[258,258.2],
              "test2":[256,262],
              "CONUS":[235,295],
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
              "ESE":[275,295]} #eastern south east

region_lat = {"test":[35.1,35.3],
              "test2":[34,38],
              "CONUS":[25,50],
              "WCN":[40,50], #west coast north
              "WCN":[25,40], #west coast south
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
#Import Data - load in all files
#########################################################################################
window = 30 #change if calculated a different window

dirname= '/scratch/bpuxley/SPI_30day'

pathfiles = []
for i in Regions:
        filename = 'SPI_%_%s.nc'%(window,i)
        pathfile = os.path.join(dirname, filename)
        pathfiles.append(pathfile)

datasets = [xr.open_dataset(f) for f in pathfiles]
print('Read in Data')

#########################################################################################
#Turn lats, lons to meshgrid and get them ready for plotting
#########################################################################################
lon_WCN, lat_WCN = np.meshgrid(datasets['df_spi_WCN'].lon.values, datasets['df_spi_WCN'].lat.values)
lon_WCS, lat_WCS = np.meshgrid(datasets['df_spi_WCS'].lon.values, datasets['df_spi_WCS'].lat.values)
lon_MWN, lat_MWN = np.meshgrid(datasets['df_spi_MWN'].lon.values, datasets['df_spi_MWN'].lat.values)
lon_MWC, lat_MWC = np.meshgrid(datasets['df_spi_MWC'].lon.values, datasets['df_spi_MWC'].lat.values)
lon_MWS, lat_MWS = np.meshgrid(datasets['df_spi_MWS'].lon.values, datasets['df_spi_MWS'].lat.values)
lon_NGP, lat_NGP = np.meshgrid(datasets['df_spi_NGP'].lon.values, datasets['df_spi_NGP'].lat.values)
lon_SGP, lat_SGP = np.meshgrid(datasets['df_spi_SGP'].lon.values, datasets['df_spi_SGP'].lat.values)
lon_NGL, lat_NGL = np.meshgrid(datasets['df_spi_NGL'].lon.values, datasets['df_spi_NGL'].lat.values)
lon_SGL, lat_SGL = np.meshgrid(datasets['df_spi_SGL'].lon.values, datasets['df_spi_SGL'].lat.values)
lon_NNE, lat_NNE = np.meshgrid(datasets['df_spi_NNE'].lon.values, datasets['df_spi_NNE'].lat.values)
lon_SNE, lat_SNE = np.meshgrid(datasets['df_spi_SNE'].lon.values, datasets['df_spi_SNE'].lat.values)
lon_WSE, lat_WSE = np.meshgrid(datasets['df_spi_WSE'].lon.values, datasets['df_spi_WSE'].lat.values)
lon_ESE, lat_ESE = np.meshgrid(datasets['df_spi_ESE'].lon.values, datasets['df_spi_ESE'].lat.values)

#########################################################################################
#Calculate the average SPI over the period at each grid point
#########################################################################################
spi_avg = {}
spi_max = {}
spi_min = {}

for key, dataset in datasets.items():
        print(key)
        spi = dataset.spi_30day[29:,:,:]
        m,o,p = spi.shape

        spi_sum = np.nansum(spi,axis=0)

        spi_avg[f'{key}'] = spi_sum/m

        spi_max[f'{key}'] = np.nanmax(spi, axis=0)
        spi_min[f'{key}'] = np.nanmin(spi, axis=0)

#########################################################################################
#Plot the SPI data
#########################################################################################
fig = plt.figure(figsize = (10,7), dpi = 300, tight_layout =True)

# First subplot with the SPI mean
ax1 = fig.add_subplot(311, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs1 = plt.contourf(lon_WCN, lat_WCN, spi_avg['df_spi_WCN'], transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-1, 1, 0.1), cmap = 'BrBG')
cs2 = plt.contourf(lon_WCS, lat_WCS, spi_avg['df_spi_WCS'], transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-1, 1, 0.1), cmap = 'BrBG')
cs3 = plt.contourf(lon_MWN, lat_MWN, spi_avg['df_spi_MWN'], transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-1, 1, 0.1), cmap = 'BrBG')
cs4 = plt.contourf(lon_MWC, lat_MWC, spi_avg['df_spi_MWC'], transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-1, 1, 0.1), cmap = 'BrBG')
cs5 = plt.contourf(lon_MWS, lat_MWS, spi_avg['df_spi_MWS'], transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-1, 1, 0.1), cmap = 'BrBG')
cs6 = plt.contourf(lon_NNE, lat_NNE, spi_avg['df_spi_NNE'], transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-1, 1, 0.1), cmap = 'BrBG')
cs7 = plt.contourf(lon_NGL, lat_NGL, spi_avg['df_spi_NGL'], transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-1, 1, 0.1), cmap = 'BrBG')
cs8 = plt.contourf(lon_SGL, lat_SGL, spi_avg['df_spi_SGL'], transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-1, 1, 0.1), cmap = 'BrBG')
cs9 = plt.contourf(lon_ESE, lat_ESE, spi_avg['df_spi_ESE'], transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-1, 1, 0.1), cmap = 'BrBG')
cs10 = plt.contourf(lon_WSE, lat_WSE, spi_avg['df_spi_WSE'], transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-1, 1, 0.1), cmap = 'BrBG')
cs11 = plt.contourf(lon_SGP, lat_SGP, spi_avg['df_spi_SGP'], transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-1, 1, 0.1), cmap = 'BrBG')
cs12 = plt.contourf(lon_NGP, lat_NGP, spi_avg['df_spi_NGP'], transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-1, 1, 0.1), cmap = 'BrBG')
cs13 = plt.contourf(lon_SNE, lat_SNE, spi_avg['df_spi_SNE'], transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-1, 1, 0.1), cmap = 'BrBG')

fig.colorbar(cs1, ax=ax1, orientation='horizontal', pad=0.05)

plt.title('SPI Average')

# Second subplot with the spi max
ax2 = fig.add_subplot(312, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linewidth=1)
ax2.add_feature(cfeature.STATES, edgecolor='black')

ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs1 = plt.contourf(lon_WCN, lat_WCN, spi_max['df_spi_WCN'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 5, 0.1), cmap = 'YlGnBu')
cs2 = plt.contourf(lon_WCS, lat_WCS, spi_max['df_spi_WCS'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 5, 0.1), cmap = 'YlGnBu')
cs3 = plt.contourf(lon_MWN, lat_MWN, spi_max['df_spi_MWN'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 5, 0.1), cmap = 'YlGnBu')
cs4 = plt.contourf(lon_MWC, lat_MWC, spi_max['df_spi_MWC'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 5, 0.1), cmap = 'YlGnBu')
cs4 = plt.contourf(lon_MWS, lat_MWS, spi_max['df_spi_MWS'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 5, 0.1), cmap = 'YlGnBu')
cs6 = plt.contourf(lon_NNE, lat_NNE, spi_max['df_spi_NNE'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 5, 0.1), cmap = 'YlGnBu')
cs7 = plt.contourf(lon_NGL, lat_NGL, spi_max['df_spi_NGL'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 5, 0.1), cmap = 'YlGnBu')
cs8 = plt.contourf(lon_SGL, lat_SGL, spi_max['df_spi_SGL'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 5, 0.1), cmap = 'YlGnBu')
cs9 = plt.contourf(lon_ESE, lat_ESE, spi_max['df_spi_ESE'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 5, 0.1), cmap = 'YlGnBu')
cs10 = plt.contourf(lon_WSE, lat_WSE, spi_max['df_spi_WSE'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 5, 0.1), cmap = 'YlGnBu')
cs11 = plt.contourf(lon_SGP, lat_SGP, spi_max['df_spi_SGP'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 5, 0.1), cmap = 'YlGnBu')
cs12 = plt.contourf(lon_NGP, lat_NGP, spi_max['df_spi_NGP'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 5, 0.1), cmap = 'YlGnBu')
cs13 = plt.contourf(lon_SNE, lat_SNE, spi_max['df_spi_SNE'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 5, 0.1), cmap = 'YlGnBu')

fig.colorbar(cs1, ax=ax2, orientation='horizontal', pad=0.05)

plt.title('SPI Max')

# Third subplot with the spi min
ax3 = fig.add_subplot(313, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax3.add_feature(cfeature.COASTLINE)
ax3.add_feature(cfeature.BORDERS, linewidth=1)
ax3.add_feature(cfeature.STATES, edgecolor='black')

ax3.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs1 = plt.contourf(lon_WCN, lat_WCN, spi_min['df_spi_WCN'], transform=ccrs.PlateCarree(), extend = "min", levels=np.arange(-5, 0, 0.1), cmap = 'copper')
cs2 = plt.contourf(lon_WCS, lat_WCS, spi_min['df_spi_WCS'], transform=ccrs.PlateCarree(), extend = "min", levels=np.arange(-5, 0, 0.1), cmap = 'copper')
cs3 = plt.contourf(lon_MWN, lat_MWN, spi_min['df_spi_MWN'], transform=ccrs.PlateCarree(), extend = "main", levels=np.arange(-5, 0, 0.1), cmap = 'copper')
cs4 = plt.contourf(lon_MWC, lat_MWC, spi_min['df_spi_MWC'], transform=ccrs.PlateCarree(), extend = "min", levels=np.arange(-5, 0, 0.1), cmap = 'copper')
cs5 = plt.contourf(lon_MWS, lat_MWS, spi_min['df_spi_MWS'], transform=ccrs.PlateCarree(), extend = "min", levels=np.arange(-5, 0, 0.1), cmap = 'copper')
cs6 = plt.contourf(lon_NNE, lat_NNE, spi_min['df_spi_NNE'], transform=ccrs.PlateCarree(), extend = "min", levels=np.arange(-5, 0, 0.1), cmap = 'copper')
cs7 = plt.contourf(lon_NGL, lat_NGL, spi_min['df_spi_NGL'], transform=ccrs.PlateCarree(), extend = "min", levels=np.arange(-5, 0, 0.1), cmap = 'copper')
cs8 = plt.contourf(lon_SGL, lat_SGL, spi_min['df_spi_SGL'], transform=ccrs.PlateCarree(), extend = "min", levels=np.arange(-5, 0, 0.1), cmap = 'copper')
cs9 = plt.contourf(lon_ESE, lat_ESE, spi_min['df_spi_ESE'], transform=ccrs.PlateCarree(), extend = "min", levels=np.arange(-5, 0, 0.1), cmap = 'copper')
cs10 = plt.contourf(lon_WSE, lat_WSE, spi_min['df_spi_WSE'], transform=ccrs.PlateCarree(), extend = "min", levels=np.arange(-5, 0, 0.1), cmap = 'copper')
cs11 = plt.contourf(lon_SGP, lat_SGP, spi_min['df_spi_SGP'], transform=ccrs.PlateCarree(), extend = "min", levels=np.arange(-5, 0, 0.1), cmap = 'copper')
cs12 = plt.contourf(lon_NGP, lat_NGP, spi_min['df_spi_NGP'], transform=ccrs.PlateCarree(), extend = "min", levels=np.arange(-5, 0, 0.1), cmap = 'copper')
cs13 = plt.contourf(lon_SNE, lat_SNE, spi_min['df_spi_SNE'], transform=ccrs.PlateCarree(), extend = "min", levels=np.arange(-5, 0, 0.1), cmap = 'copper')

fig.colorbar(cs1, ax=ax3, orientation='horizontal', pad=0.05)

plt.title('SPI Min')

plt.savefig('/scratch/bpuxley/Plots/spi_CONUS', bbox_inches = 'tight', pad_inches = 0.1)



