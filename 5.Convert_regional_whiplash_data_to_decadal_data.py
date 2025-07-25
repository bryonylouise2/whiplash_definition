#########################################################################################
## Script to convert regional whiplash data into decadal periods for later analysis
## Bryony Louise
## Last Edited: Friday, July 25th, 2025
## Input: regional whiplash data
## Output: multiple netCDF files of identified whiplash occurrences at a ~ 6 km grid resolution from
## 1915 to 2020, split into 10-year decade periods. If files were regional, they have been
## combined into CONUS-wide.
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
# Convert regional files into decadal periods 
#########################################################################################
Regions = {"WCN", "WCS", "MWN", "MWC", "MWS", "NGP", "SGP", "NGL", "SGL", "NNE", "SNE", "ESE", "WSE"}
time_periods = {'1915_1924','1925_1934','1935_1944','1945_1954', '1955_1964','1965_1974','1975_1984','1985_1994','1995_2004','2005_2014','2015_2020'}

dirname= '/data2/bpuxley/Whiplash/Regional_data'

pathfiles = []
for i in Regions:
        filename = 'whiplashes_%s.nc'%(i)
        pathfile = os.path.join(dirname, filename)
        pathfiles.append(pathfile)

datasets = [xr.open_dataset(f) for f in pathfiles]
print('Read in Data')

#Split whiplash data into 10-year periods
for period in sorted(time_periods):
	print(period)
	data = [ds.sel(time=slice(period[:4]+"-01-01",period[5:]+"-12-31")) for ds in datasets]
	
	print(period+': Started')
	combined = xr.combine_by_coords(data)
	print(period+': Ended')
	
	print('Saving '+period+': Started')
	combined.to_netcdf('/data2/bpuxley/droughts_and_pluvials/droughts_and_pluvials_%s.nc'%(period))
	print('Saving '+period+': Ended')

	del(combined)
