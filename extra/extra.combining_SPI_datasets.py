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

print('1915-1924: Started')
combined = xr.combine_by_coords(time_1915_1924)
print('1915-1924: Ended')

print('Saving 1915-1924: Started')
combined.to_netcdf('/scratch/bpuxley/SPI_30day/spi_1915_1924.nc')
print('Saving 1915-1924: Ended')

del(combined)

print('1925-1934: Started')
combined = xr.combine_by_coords(time_1925_1934)
print('1925-1934: Ended')

print('Saving 1925-1934: Started')
combined.to_netcdf('/scratch/bpuxley/SPI_30day/spi_1925_1934.nc')
print('Saving 1925-1934: Ended')

del(combined)

print('1935-1944: Started')
combined = xr.combine_by_coords(time_1935_1944)
print('1935-1944: Ended')

print('Saving 1935-1944: Started')
combined.to_netcdf('/scratch/bpuxley/SPI_30day/spi_1935_1944.nc')
print('Saving 1935-1944: Ended')

del(combined)

print('1945-1954: Started')
combined = xr.combine_by_coords(time_1945_1954)
print('1945-1954: Ended')

print('Saving 1945-1954: Started')
combined.to_netcdf('/scratch/bpuxley/SPI_30day/spi_1945_1954.nc')
print('Saving 1945-1954: Ended')

del(combined)

print('1955-1964: Started')
combined = xr.combine_by_coords(time_1955_1964)
print('1955-1964: Ended')

print('Saving 1955-1964: Started')
combined.to_netcdf('/scratch/bpuxley/SPI_30day/spi_1955_1964.nc')
print('Saving 1955-1964: Ended')

del(combined)

print('1965-1974: Started')
combined = xr.combine_by_coords(time_1965_1974)
print('1965-1974: Ended')

print('Saving 1965-1974: Started')
combined.to_netcdf('/scratch/bpuxley/SPI_30day/spi_1965_1974.nc')
print('Saving 1965-1974: Ended')

del(combined)

print('1975-1984: Started')
combined = xr.combine_by_coords(time_1975_1984)
print('1975-1984: Ended')

print('Saving 1975-1984: Started')
combined.to_netcdf('/scratch/bpuxley/SPI_30day/spi_1975_1984.nc')
print('Saving 1975-1984: Ended')

del(combined)

print('1985-1994: Started')
combined = xr.combine_by_coords(time_1985_1994)
print('1985-1994: Ended')

print('Saving 1985-1994: Started')
combined.to_netcdf('/scratch/bpuxley/SPI_30day/spi_1985_1994.nc')
print('Saving 1985-1994: Ended')

del(combined)

print('1995-2004: Started')
combined = xr.combine_by_coords(time_1995_2004)
print('1995-2004: Ended')

print('Saving 1995-2004: Started')
combined.to_netcdf('/scratch/bpuxley/SPI_30day/spi_1995_2004.nc')
print('Saving 1995-2004: Ended')

del(combined)

print('2005-2014: Started')
combined = xr.combine_by_coords(time_2005_2014)
print('2005-2014: Ended')

print('Saving 2005-2014: Started')
combined.to_netcdf('/scratch/bpuxley/SPI_30day/spi_2005_2014.nc')
print('Saving 2005-2014: Ended')

del(combined)

print('2015-2020: Started')
combined = xr.combine_by_coords(time_2015_2020)
print('2015-2020: Ended')

print('Saving 2015-2020: Started')
combined.to_netcdf('/scratch/bpuxley/SPI_30day/spi_2015_2020.nc')
print('Saving 2015-2020: Ended')

del(combined)
