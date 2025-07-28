#########################################################################################
## This script calculates the area of all polygons that come out of the KDE process.
## Bryony Louise
## Last Edited: Monday, July 28th, 2025
## Input: Decadal density files.
## Output: Decadal csv files (1 for drought-to-pluvial, 1 for pluvial-to-drought) that have 
## a list of all potential events throughout the time frame, including: Drought Date, 
## Pluvial Date, Whiplash Date, Area (km2), and polygon geometry. 
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
from matplotlib.colors import ListedColormap
from matplotlib.path import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import spei as si
import pandas as pd
import scipy.stats as scs
import os

#########################################################################################
# Import Functions
#########################################################################################
import functions

#########################################################################################
# Import Data
#########################################################################################
years = ['1915_1924', '1925_1934', '1935_1944', '1945_1954', '1955_1964', '1965_1974', 
			'1975_1984', '1985_1994', '1995_2004', '2005_2014', '2015_2020']

#Choose which file to look at
data_slice = years[0] #choose which years to look at

dirname = '/data2/bpuxley/Density/density_%s.nc'%(data_slice)

all_densities = xr.open_dataset(dirname)

#Split into DP and PD
DP_densities = all_densities.DP_density
PD_densities = all_densities.PD_density

m,o,p = DP_densities.shape
dates = all_densities.time.values

grid_lons = all_densities.lon.values
grid_lats = all_densities.lat.values

print('Read in Data')
#########################################################################################
# Calculate the area of all polygons
#########################################################################################
ISOPLETH_DP = 0.4878 #the 99th percentile of all DP density fields
ISOPLETH_PD = 0.4783 #the 99th percentile of all PD density fields

#Drought-to-Pluvial
areas_DP = []
polys_DP = []
kept_dates_DP = []

print('Drought-to-Pluvial loop: started')
for i in tqdm(range(29,m)): #loop through the timeseries
	DP_field = DP_densities[i,:,:].values
	if np.any(DP_field):  #only run code if at least one of the values is 1
            a, p = functions.calc_area(grid_lons, grid_lats, DP_field,
                                     isopleth=ISOPLETH_DP, area_threshold=0)
            areas_DP.extend(a)
            polys_DP.extend(p)
            kept_dates_DP.extend([dates[i]] * len(a))
print('Drought-to-Pluvial loop: ended')

data_for_file_dp = {
    'Drought_Date': pd.DatetimeIndex([i - pd.DateOffset(days=29) for i in kept_dates_DP]),
    'Pluvial_Date': pd.DatetimeIndex([i + pd.DateOffset(days=1) for i in kept_dates_DP]),
    'Whiplash_Date': pd.DatetimeIndex([i for i in kept_dates_DP]),
    'Area': areas_DP,
    'geometry': polys_DP 
    }

print('Drought-to-Pluvial: saving file')
df_dp = pd.DataFrame(data_for_file_dp)
df_dp.to_csv(f'/data2/bpuxley/Databases/potential_events_DP_%s.csv'%(data_slice), index=False)
print('Drought-to-Pluvial: file saved')

#Pluvial-to-Drought
areas_PD = []
polys_PD = []
kept_dates_PD = []

print('Pluvial-to-Drought loop: started')
for i in tqdm(range(0,m)): #loop through the timeseries
	PD_field = PD_densities[i,:,:].values
	if np.any(PD_field):  #only run code if at least one of the values is 1
            a, p = functions.calc_area(grid_lons, grid_lats, PD_field,
                                     isopleth=ISOPLETH_PD, area_threshold=0)
            areas_PD.extend(a)
            polys_PD.extend(p)
            kept_dates_PD.extend([dates[i]] * len(a))
print('Drought-to-Pluvial loop: ended')

data_for_file_pd = {
    'Pluvial_Date': pd.DatetimeIndex([i - pd.DateOffset(days=29) for i in kept_dates_PD]),
    'Drought_Date': pd.DatetimeIndex([i + pd.DateOffset(days=1) for i in kept_dates_PD]),
    'Whiplash_Date': pd.DatetimeIndex([i for i in kept_dates_PD]),
    'Area': areas_PD,
    'geometry': polys_PD 
    }

print('Pluvial-to-Drought: saving file')
df_pd = pd.DataFrame(data_for_file_pd)
df_pd.to_csv(f'/data2/bpuxley/Databases/potential_events_PD_%s.csv'%(data_slice), index=False)
print('Pluvial-to-Drought:file saved')
