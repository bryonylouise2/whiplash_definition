#########################################################################################
## Consider the spatial continuity of grid points experiencing a whiplash event to determine events
## Bryony Louise
## Last Edited: Friday February 28th 2025 
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
#Import Data - load in all files
#########################################################################################
years = ['1915_1924', '1925_1934', '1935_1944', '1945_1954', '1955_1964', '1965_1974', 
			'1975_1984', '1985_1994', '1995_2004', '2005_2014', '2015_2020']

#Whiplash Points from 4.Whiplash_Identification
dirname= '/scratch/bpuxley/Whiplash'

pathfiles = []

for i in years:
	filename = 'whiplashes_%s.nc'%(i)
	pathfile = os.path.join(dirname, filename)
	pathfiles.append(pathfile)

whiplashes = xr.open_mfdataset(pathfiles, combine='by_coords')

#Split into managable time chunks for data processing
data_slice = years[0] #choose which years to look at
start_year = years[0][0:4]
end_year = years[0][5:9]
dataset_DP = whiplashes.DP_whiplashes.sel(time=slice(start_year +'-01-01', end_year +'-12-31'))
dataset_PD = whiplashes.PD_whiplashes.sel(time=slice(start_year +'-01-01', end_year +'-12-31'))

lon, lat = np.meshgrid(dataset_DP.lon.values, dataset_DP.lat.values)
m,o,p = dataset_DP.shape

print('Read in Data')
print('Time: ' + str(m))
print('Lat: ' + str(o))
print('Lon: ' + str(p))

#########################################################################################
#Determine Spatial Consistency
#########################################################################################
density_DP = np.zeros((m,o,p))
density_PD = np.zeros((m,o,p))

for i in tqdm(range(0, m)): #loop through the timeseries: when looking at the beginning of the time series turn 0 to 29 and when looking at the end of the series turn m to m-30 (first and last 30 values will be all nans)
        DP_field = dataset_DP[i,:,:].values
        if np.any(~np.isnan(DP_field)): #only run KDE code if at least one of the values is 1
                density_DP[i, :, :] = functions.kde(orig_lon=dataset_DP.lon.values,
                                                                orig_lat=dataset_DP.lat.values,
                                                                grid_lon=dataset_DP.lon.values,
                                                                grid_lat=dataset_DP.lat.values,
                                                                extreme=DP_field,
                                                                bandwidth=0.025,
                                                                        )

for i in tqdm(range(0, m)): #loop through the timeseries: when looking at the beginning of the time series turn 0 to 29 and when looking at the end of the series turn m to m-30 (first and last 30 values will be all nans)
        PD_field = dataset_PD[i,:,:].values
        if np.any(~np.isnan(PD_field)): #only run KDE code if at least one of the values is 1
                density_PD[i, :, :] = functions.kde(orig_lon=dataset_PD.lon.values,
                                                                orig_lat=dataset_PD.lat.values,
                                                                grid_lon=dataset_PD.lon.values,
                                                                grid_lat=dataset_PD.lat.values,
                                                                extreme=PD_field,
                                                                bandwidth=0.025,
                                                                        )


#########################################################################################
#Save out the density file 
#########################################################################################
time = dataset_DP.time
lats = dataset_DP.lat
lons = dataset_DP.lon
attrs = dataset_DP.attrs

dimensions=['time','lat','lon']
coords = {
                        'time': time,
                        'lat': lats,
                        'lon': lons
                        }
#attributes = {'sources': 'Livneh et al., 2013 & PRISM, 2004', 'references': 'http://www.esrl.noaa.gov/psd/data/gridded/data.livneh.html','}

DP_density = xr.DataArray(density_DP, coords, dims=dimensions, name='Drought to Pluvial')
PD_density = xr.DataArray(density_PD, coords, dims=dimensions, name='Pluvial to Drought')
density_dataset = xr.Dataset({"DP_density":DP_density, "PD_density":PD_density})

#Remember to change filename to whatever years have been calculated
density_dataset.to_netcdf('/scratch/bpuxley/Density/density_%s.nc'%(data_slice))
print('density file saved')
