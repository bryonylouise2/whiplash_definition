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
dirname= '/data2/bpuxley/SPI_30day'

pathfiles = []
for i in Regions:
        filename = 'SPI30_%s.nc'%(i)
        pathfile = os.path.join(dirname, filename)
        pathfiles.append(pathfile)


datasets = [xr.open_dataset(f) for f in pathfiles]
print('Read in Data')

#Split spi data in 10 year time periods
time_1915_1924 = [ds.sel(time=slice("1915-01-01","1924-12-31")) for ds in datasets]
time_1925_1934 = [ds.sel(time=slice("1925-01-01","1934-12-31")) for ds in datasets]
time_1935_1944 = [ds.sel(time=slice("1935-01-01","1944-12-31")) for ds in datasets]
time_1945_1954 = [ds.sel(time=slice("1945-01-01","1954-12-31")) for ds in datasets]
time_1955_1964 = [ds.sel(time=slice("1955-01-01","1964-12-31")) for ds in datasets]
time_1965_1974 = [ds.sel(time=slice("1965-01-01","1974-12-31")) for ds in datasets]
time_1975_1984 = [ds.sel(time=slice("1975-01-01","1984-12-31")) for ds in datasets]
time_1985_1994 = [ds.sel(time=slice("1985-01-01","1994-12-31")) for ds in datasets]
time_1995_2004 = [ds.sel(time=slice("1995-01-01","2004-12-31")) for ds in datasets]
time_2005_2014 = [ds.sel(time=slice("2005-01-01","2014-12-31")) for ds in datasets]
time_2015_2020 = [ds.sel(time=slice("2015-01-01","2020-12-31")) for ds in datasets]

combined = xr.combine_by_coords(datasets)
print('Combined Datafiles')

combined.to_netcdf('scratch/bpuxley/SPI_30day/spi_CONUS.nc')
print('Saved file to netcdf')
