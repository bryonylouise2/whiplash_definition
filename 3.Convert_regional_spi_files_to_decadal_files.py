#########################################################################################
## Script to convert the regional SPI.nc files for each region into decadal files for later analysis
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
#Convert regional files into decadal time periods 
#########################################################################################
Regions = {"WCN", "WCS", "MWN", "MWC", "MWS", "NGP", "SGP", "NGL", "SGL", "NNE", "SNE", "ESE", "WSE"}
time_periods = {'1915_1924','1925_1934','1935_1944','1945_1954', '1955_1964','1965_1974','1975_1984','1985_1994','1995_2004','2005_2014','2015_2020'}

dirname= '/scratch/bpuxley/SPI_30day'

pathfiles = []
for i in Regions:
        filename = 'SPI30_%s.nc'%(i)
        pathfile = os.path.join(dirname, filename)
        pathfiles.append(pathfile)

datasets = [xr.open_dataset(f) for f in pathfiles]
print('Read in Data')

#Split data in 10 year time periods
for period in sorted(time_periods):
	print(period)
	data = [ds.sel(time=slice(period[:4]+"-01-01",period[5:]+"-12-31")) for ds in datasets]
	
	print(period+': Started')
	combined = xr.combine_by_coords(data)
	print(period+': Ended')
	
	print('Saving '+period+': Started')
	combined.to_netcdf('/scratch/bpuxley/SPI_30day/spi_%s.nc'%(period))
	print('Saving '+period+': Ended')

	del(combined)
