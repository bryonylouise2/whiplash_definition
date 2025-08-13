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
#Time Periods
#########################################################################################
#Time Periods
Years = {"1915_1924", "1925_1934", "1935_1944", "1945_1954", "1955_1964", "1965_1974",
                        "1975_1984", "1985_1994", "1995_2004", "2005_2014", "2015_2020"}

#########################################################################################
#Import Data - load in all files
#########################################################################################
#drought_dirname = '/scratch/bpuxley/droughts_and_pluvials/density/density_droughts_*.nc'
pluvial_dirname = '/scratch/bpuxley/droughts_and_pluvials/density/density_pluvials_*.nc'

#drought_densities = xr.open_mfdataset(drought_dirname, combine='by_coords')
pluvial_densities = xr.open_mfdataset(pluvial_dirname, combine='by_coords')

#Save file out
#all_densities.to_netcdf('/scratch/bpuxley/Density/density_1915_2020.nc')

#Split into DP and PD
#drought_densities = drought_densities.drought_density
pluvial_densities = pluvial_densities.pluvial_density

#Calculate the 99th percentile of density files to draw the polygon
#drought_perc = np.percentile(drought_densities, q=99)
pluvial_perc = np.percentile(pluvial_densities, q=99)
print(f'the 99th percentile for Drought Events is 0.8921, for Pluvial Events is {pluvial_perc}, and they were calculated at {datetime.now()}')
