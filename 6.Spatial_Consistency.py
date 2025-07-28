#########################################################################################
## Using Kernel Density Estimation; consider the spatial continuity of grid points 
## experiencing a whiplash event to determine events.
## Bryony Louise
## Last Edited: Friday, July 25th, 2025 
## Input: Decadal whiplash occurrence files.
## Output: netCDF files of the density of whiplash grid points at each day throughout the
## period. Higher values represent a greater density of whiplash occurrences.
#########################################################################################
# Import Required Modules
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
# Import Functions
#########################################################################################
import functions

#########################################################################################
# Import Data - load in all files
#########################################################################################
time_periods = {'1915_1924', '1925_1934', '1935_1944', '1945_1954', '1955_1964', '1965_1974', 
			'1975_1984', '1985_1994', '1995_2004', '2005_2014', '2015_2020'}

#Whiplash Points from 5.Convert_regional_whiplash_data_into_decadal_periods.py
dirname= '/scratch/bpuxley/Whiplash'

pathfiles = []

for period in sorted(time_periods):
	print(period)

        ################################
        # Read in data
        ################################
	filename = 'whiplashes_%s.nc'%(period)
	pathfile = os.path.join(dirname, filename)
        df = xr.open_dataset(pathfile)
        print('Read in Data')

        m,o,p = len(df.time), len(df.lat), len(df.lon) #length of data #mo,o,p: time, lat, lon 
        lon, lat = np.meshgrid(df.lon.values, df.lat.values) #create a meshgrid of lat,lon values

        print('Time: '+ str(m))
        print('Lat: '+ str(o))
        print('Lon: '+ str(p))

        DP_dataset = df.DP_whiplashes
        PD_dataset = df.PD_whiplashes
        
	################################
        # Determine Spatial Consistency
        ################################
        density_DP = np.zeros((m,o,p))
        density_PD = np.zeros((m,o,p))

        start=time.time()
        print('Drought-to-Pluvial: Started')
        for i in tqdm(range(0, m)): #loop through the timeseries: when looking at the beginning of the time series, turn 0 to 2,9 and when looking at the end of the series, turn m to m-30 (first and last 30 values will be all nans)
                DP_field = DP_dataset[i,:,:].values
                if np.any(~np.isnan(DP_field)): #only run KDE code if at least one of the values is 1
                        density_DP[i, :, :] = functions.kde(orig_lon=DP_dataset.lon.values,
                                                                orig_lat=DP_dataset.lat.values,
                                                                grid_lon=DP_dataset.lon.values,
                                                                grid_lat=DP_dataset.lat.values,
                                                                extreme=DP_field,
                                                                bandwidth=0.025,
                                                                                )
        print('Drought-to-Pluvial: Ended')
        end=time.time()
        time_taken = (end-start)/3600
        print('Time Taken: '+str(time_taken))

        start=time.time()
        print('Pluvial-to-Drought: Started')
        for i in tqdm(range(0, m)): #loop through the timeseries: when looking at the beginning of the time series turn 0 to 29 and when looking at the end of the series turn m to m-30 (first and last 30 values will be all nans)
                PD_field = PD_dataset[i,:,:].values
                if np.any(~np.isnan(PD_field)): #only run KDE code if at least one of the values is 1
                        density_PD[i, :, :] = functions.kde(orig_lon=PD_dataset.lon.values,
                                                                orig_lat=PD_dataset.lat.values,
                                                                grid_lon=PD_dataset.lon.values,
                                                                grid_lat=PD_dataset.lat.values,
                                                                extreme=PD_field,
                                                                bandwidth=0.025,
                                                                                )
        print('Pluvial-to-Drought: Ended')
        end=time.time()
        time_taken=(end-start)/3600
        print('Time Taken: '+str(time_taken))                                                               

        ################################
        #Save out the density file 
        ##########i######################
        time_frame = DP_dataset.time
        lats = DP_dataset.lat
        lons = DP_dataset.lon

        dimensions=['time', 'lat', 'lon']
        coords = {
                'time': time_frame,
                'lat': lats,
                'lon': lons
                        }

       	DP_density = xr.DataArray(density_DP, coords, dims=dimensions, name='Drought-to-Pluvial')
        PD_density = xr.DataArray(density_PD, coords, dims=dimensions, name='Pluvial-to-Drought')
        density_dataset = xr.Dataset({"DP_density":DP_density, "PD_density":PD_density})

        density_dataset.to_netcdf('/scratch/bpuxley/density/density_%s.nc'%(period))
        print('density file saved')
        print('Period: '+period+' completed')
