#########################################################################################
## Script to combine the multiple SPI.nc files for each region into one 
## Bryony Louise
## Last Edited: Thursday October 17th 2024 
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
import gzip

#########################################################################################
#Regions For Analysis
#########################################################################################
#Choose Region
Regions = {"WCN", "WCS", "MWN", "MWC", "MWS", "NGP", "SGP", "NGL", "SGL", "SNE", "NNE", "WSE", "ESE"}
              
#########################################################################################
#Import Data
#########################################################################################
#########################################################################################
#Import Data - load in all files
#########################################################################################
dirname= '/scratch/bpuxley/SPI_30day'

pathfiles = []
for i in Regions:
        filename = 'SPI30_%s.nc'%(i)
        pathfile = os.path.join(dirname, filename)
        pathfiles.append(pathfile)


datasets = [xr.open_dataset(f) for f in pathfiles]
print('Read in Data')

#Split spi data in 10 year time periods
time_1915_1924 = [ds.isel(time=slice(0,19358)) for ds in datasets]
time_1925_1934 = [ds.isel(time=slice(0,19358)) for ds in datasets]
time_1935_1944 = [ds.isel(time=slice(0,19358)) for ds in datasets]
time_1945_1954 = [ds.isel(time=slice(0,19358)) for ds in datasets]
time_1955_1964 = [ds.isel(time=slice(0,19358)) for ds in datasets]
time_1965_1974 = [ds.isel(time=slice(0,19358)) for ds in datasets]
time_1975_1984 = [ds.isel(time=slice(0,19358)) for ds in datasets]
time_1985_1994 = [ds.isel(time=slice(0,19358)) for ds in datasets]
time_1995_2004 = [ds.isel(time=slice(0,19358)) for ds in datasets]
time_2005_2014 = [ds.isel(time=slice(0,19358)) for ds in datasets]
time_2015_2020 = [ds.isel(time=slice(0,19358)) for ds in datasets]

combined = xr.combine_by_coords(datasets)
print('Combined Datafiles')

combined.to_netcdf('scratch/bpuxley/SPI_30day/spi_CONUS.nc')
print('Saved file to netcdf')
