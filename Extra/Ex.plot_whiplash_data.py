#########################################################################################
## Plot the number of whiplash occurrences at each grid point throughout the timeframe
## Bryony Louise Puxley
## Last Edited: Friday, August 15th, 2025 
## Input: regional whiplash data
## Output: A PNG of the number of whiplash occurrences throughout the period across the CONUS.
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
Regions = { "MWN", "MWC", "MWS", "NGP", "SGP", "NGL", "SGL", "NNE", "SNE", "WSE", "ESE"}

region_lon = {"test":[277,277.2],
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
              "WSE":[265,279], #western south east
              "ESE":[279,295]} #eastern south east

region_lat = {"test":[35.1,35.3], 
              "CONUS":[25,50],
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

#########################################################################################
#Import Data - load in all files
#########################################################################################
dirname= '/scratch/bpuxley/Whiplash'

pathfiles = []
for i in Regions:
        filename = 'whiplashes_%s.nc'%(i)
        pathfile = os.path.join(dirname, filename)
        pathfiles.append(pathfile)
  
datasets = [xr.open_dataset(f) for f in pathfiles]
print('Read in Data')

#########################################################################################
#Turn lats, lons to meshgrid and get them ready for plotting
#########################################################################################
lon_WCN, lat_WCN = np.meshgrid(datasets['df_whiplash_WCN'].lon.values, datasets['df_whiplash_WCN'].lat.values)
lon_WCS, lat_WCS = np.meshgrid(datasets['df_whiplash_WCS'].lon.values, datasets['df_whiplash_WCS'].lat.values)
lon_MWN, lat_MWN = np.meshgrid(datasets['df_whiplash_MWN'].lon.values, datasets['df_whiplash_MWN'].lat.values)
lon_MWC, lat_MWC = np.meshgrid(datasets['df_whiplash_MWC'].lon.values, datasets['df_whiplash_MWC'].lat.values)
lon_MWS, lat_MWS = np.meshgrid(datasets['df_whiplash_MWS'].lon.values, datasets['df_whiplash_MWS'].lat.values)
lon_NGP, lat_NGP = np.meshgrid(datasets['df_whiplash_NGP'].lon.values, datasets['df_whiplash_NGP'].lat.values)
lon_SGP, lat_SGP = np.meshgrid(datasets['df_whiplash_SGP'].lon.values, datasets['df_whiplash_SGP'].lat.values)
lon_NGL, lat_NGL = np.meshgrid(datasets['df_whiplash_NGL'].lon.values, datasets['df_whiplash_NGL'].lat.values)
lon_SGL, lat_SGL = np.meshgrid(datasets['df_whiplash_SGL'].lon.values, datasets['df_whiplash_SGL'].lat.values)
lon_NNE, lat_NNE = np.meshgrid(datasets['df_whiplash_NNE'].lon.values, datasets['df_whiplash_NNE'].lat.values)
lon_SNE, lat_SNE = np.meshgrid(datasets['df_whiplash_SNE'].lon.values, datasets['df_whiplash_SNE'].lat.values)
lon_WSE, lat_WSE = np.meshgrid(datasets['df_whiplash_WSE'].lon.values, datasets['df_whiplash_WSE'].lat.values)
lon_ESE, lat_ESE = np.meshgrid(datasets['df_whiplash_ESE'].lon.values, datasets['df_whiplash_ESE'].lat.values)


#########################################################################################
#Extract the DP and PD counts for all regions
#########################################################################################
DPcount = {}
PDcount = {}

for key, dataset in datasets.items():
        print(key)
        PDcount[f'{key}'] = dataset.PD_count[:,:]
        DPcount[f'{key}'] = dataset.DP_count[:,:]

#########################################################################################
#Plot the count of whiplash events across the CONUS
#########################################################################################
fig = plt.figure(figsize = (6,7), dpi = 300, tight_layout =True)

# First subplot with the count of drought-to-pluvial whiplash number
ax1 = fig.add_subplot(211, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs1 = plt.contourf(lon_MWN, lat_MWN, DPcount['df_whiplash_MWN'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs2 = plt.contourf(lon_MWC, lat_MWC, DPcount['df_whiplash_MWC'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs3 = plt.contourf(lon_MWS, lat_MWS, DPcount['df_whiplash_MWS'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs4 = plt.contourf(lon_NGP, lat_NGP, DPcount['df_whiplash_NGP'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs5 = plt.contourf(lon_SGP, lat_SGP, DPcount['df_whiplash_SGP'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs6 = plt.contourf(lon_NGL, lat_NGL, DPcount['df_whiplash_NGL'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs7 = plt.contourf(lon_SGL, lat_SGL, DPcount['df_whiplash_SGL'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs8 = plt.contourf(lon_NNE, lat_NNE, DPcount['df_whiplash_NNE'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs9 = plt.contourf(lon_SNE, lat_SNE, DPcount['df_whiplash_SNE'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs10 = plt.contourf(lon_WSE, lat_WSE, DPcount['df_whiplash_WSE'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs11 = plt.contourf(lon_ESE, lat_ESE, DPcount['df_whiplash_ESE'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')

fig.colorbar(cs1, ax=ax1, orientation='horizontal', pad=0.05)

plt.title('Drought-to-Pluvial')

# Second subplot with thr count of pluvial-to-drought whiplash number
ax2 = fig.add_subplot(212, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linewidth=1)
ax2.add_feature(cfeature.STATES, edgecolor='black')

ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs1 = plt.contourf(lon_MWN, lat_MWN, PDcount['df_whiplash_MWN'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs2 = plt.contourf(lon_MWC, lat_MWC, PDcount['df_whiplash_MWC'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs3 = plt.contourf(lon_MWS, lat_MWS, PDcount['df_whiplash_MWS'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs4 = plt.contourf(lon_NGP, lat_NGP, PDcount['df_whiplash_NGP'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs5 = plt.contourf(lon_SGP, lat_SGP, PDcount['df_whiplash_SGP'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs6 = plt.contourf(lon_NGL, lat_NGL, PDcount['df_whiplash_NGL'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs7 = plt.contourf(lon_SGL, lat_SGL, PDcount['df_whiplash_SGL'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs8 = plt.contourf(lon_NNE, lat_NNE, PDcount['df_whiplash_NNE'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs9 = plt.contourf(lon_SNE, lat_SNE, PDcount['df_whiplash_SNE'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs10 = plt.contourf(lon_WSE, lat_WSE, PDcount['df_whiplash_WSE'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')
cs11 = plt.contourf(lon_ESE, lat_ESE, PDcount['df_whiplash_ESE'], transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 1200, 2), cmap = 'hot_r')

fig.colorbar(cs1, ax=ax2, orientation='horizontal', pad=0.05)

plt.title('Pluvial-to-Drought')

plt.savefig('/scratch/bpuxley/Plots/Whiplashes_CONUS_1915_2020', bbox_inches = 'tight', pad_inches = 0.1)

  
