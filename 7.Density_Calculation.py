#########################################################################################
## Script to combine the multiple densitys.nc files into one file 
## and calculate the 99th percentile for polygons
## Bryony Louise
## Last Edited: Tuesday November 12th 2024 
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
#Import Data - load in all files
#########################################################################################
dirname= '/data2/bpuxley/Density/*.nc'

all_densities = xr.open_mfdataset(dirname, combine='by_coords')

#Save file out
#all_densities.to_netcdf('/scratch/bpuxley/Density/density_1915_2020.nc')

#Split into DP and PD
DP_densities = all_densities.DP_density
PD_densities = all_densities.PD_density

#Calculate the 99th percentile of density files to draw the polygon
DP_perc = np.percentile(DP_densities, q=99)
PD_perc = np.percentile(PD_densities, q=99)
print(f'the 99th percentile for Drought-to-Pluvial Events is {DP_perc}, for Pluvial-to-Drought Events is {PD_perc}, and they were calculated at {datetime.now()}')
