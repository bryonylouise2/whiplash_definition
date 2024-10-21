#########################################################################################
## Script to combine the multiple whiplashes.nc files for each region into one 
## Bryony Louise
## Last Edited: Monday October 21st 2024 
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
Regions = {"MWN", "MWC", "MWS", "NGP", "SGP", "NGL", "SGL", "SNE", "WSE", "ESE"}
              
#########################################################################################
#Import Data
#########################################################################################
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

combined = xr.combine_by_coords(datasets)
print('Combined Datafiles')

combined.to_netcdf('scratch/bpuxley/Whiplash/whiplashes_CONUS.nc')
print('Saved file to netcdf')
