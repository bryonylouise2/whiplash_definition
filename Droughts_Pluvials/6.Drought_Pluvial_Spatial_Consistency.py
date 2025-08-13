#########################################################################################
## Using Kernel Density Estimation; consider the spatial continuity of grid points 
## experiencing a drought/pluvial event to determine events.
## Bryony Louise Puxley
## Last Edited: Wednesday, August 13th, 2025 
## Input: Decadal drought/pluvial occurrence files.
## Output: netCDF files of the density of drought/pluvial grid points at each day throughout 
## the period. Higher values represent a greater density of drought/pluvial occurrences.
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
#Import Functions
#########################################################################################
import functions

#########################################################################################
#Spatial Consistency
#########################################################################################
time_periods = {'1915_1924','1925_1934','1935_1944','1945_1954', '1955_1964','1965_1974','1975_1984','1985_1994','1995_2004','2005_2014','2015_2020'}

dirname= '/data2/bpuxley/droughts_and_pluvials/'

for period in sorted(time_periods):
	print(period)
	
	################################
	# Read in data
	################################
	filename = 'droughts_and_pluvials_%s.nc'%(period)	
	pathfile = os.path.join(dirname, filename)

	df = xr.open_dataset(pathfile)
	print('Read in Data')

	m,o,p = len(df.time), len(df.lat), len(df.lon) #length of data #mo,o,p: time, lat, lon 
	lon, lat = np.meshgrid(df.lon.values, df.lat.values) #create a meshgrid of lat,lon values

	print('Time: '+ str(m))
	print('Lat: '+ str(o))
	print('Lon: '+ str(p))
	
	drought_dataset = df.drought_events
	pluvial_dataset = df.pluvial_events

	################################
	# Determine Spatial Consistency
	################################
	density_drought = np.zeros((m,o,p))
	density_pluvial = np.zeros((m,o,p))
	
	start = time.time()
	print('Drought: Started')
	for i in tqdm(range(29, m)): #loop through the timeseries: when looking at the beginning of the time series turn 0 to 29 and when looking at the end of the series turn m to m-30 (first and last 30 values will be all nans)
        drought_field = drought_dataset[i,:,:].values
        if np.any(drought_field): #only run KDE code if at least one of the values is 1
                density_drought[i, :, :] = functions.kde(orig_lon=drought_dataset.lon.values,
                                                                orig_lat=drought_dataset.lat.values,
                                                                grid_lon=drought_dataset.lon.values,
                                                                grid_lat=drought_dataset.lat.values,
                                                                extreme=drought_field,
                                                                bandwidth=0.025,
                                                                        )

	print('Drought: Ended')
	end = time.time()
	
	time_taken = (end-start)/3600
	
	print('Time Taken: '+time_taken)
	
	print('Pluvial: Started')
	for i in tqdm(range(0, m)): #loop through the timeseries: when looking at the beginning of the time series turn 0 to 29 and when looking at the end of the series turn m to m-30 (first and last 30 values will be all nans)
        pluvial_field = dataset_pluvial[i,:,:].values
        if np.any(pluvial_field): #only run KDE code if at least one of the values is 1
                density_pluvial[i, :, :] = functions.kde(orig_lon=pluvial_dataset.lon.values,
                                                                orig_lat=pluvial_dataset.lat.values,
                                                                grid_lon=pluvial_dataset.lon.values,
                                                                grid_lat=pluvial_dataset.lat.values,
                                                                extreme=pluvial_field,
                                                                bandwidth=0.025,
                                                                        )
    print('Pluvial: Ended')     
    
	################################
	#Save out the density file 
	################################
	time_frame = drought_dataset.time
	lats = drought_dataset.lat
	lons = drought_dataset.lon

	dimensions=['time','lat','lon']
	coords = {
				'time': time_frame,
                'lat': lats,
                'lon': lons
                }

	drought_density = xr.DataArray(density_drought, coords, dims=dimensions, name='Drought')
	pluvial_density = xr.DataArray(density_pluvial, coords, dims=dimensions, name='Pluvial')
	density_dataset = xr.Dataset({"drought_density":drought_density, "pluvial_density":pluvial_density})

	density_dataset.to_netcdf('/data2/bpuxley/droughts_and_pluvials/density/density_%s.nc'%(period))
	print('density file saved')
                                                       
    print('Period: '+period+' completed')
