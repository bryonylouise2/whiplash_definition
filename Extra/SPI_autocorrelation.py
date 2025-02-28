#########################################################################################
## Determine the autocorrelation of SPI across the CONUS
## Bryony Louise
## Last Edited: Tuesday January 28th 2025 
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
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import spei as si
import pandas as pd
import scipy.stats as scs
import shapely.wkt
import os


#########################################################################################
#Import Functions
#########################################################################################
import functions

#########################################################################################
#Create a time-only-average file of SPI values across the CONUS (run on oscer)
#########################################################################################
'''
years = ['1915_1924', '1925_1934', '1935_1944', '1945_1954', '1955_1964', '1965_1974',
                        '1975_1984', '1985_1994', '1995_2004', '2005_2014', '2015_2020']

#SPI from 2.Calculating_SPI
dirname= '/scratch/bpuxley/SPI_30day'

pathfiles = []

for i in years:
        filename = 'spi_%s.nc'%(i)
        pathfile = os.path.join(dirname, filename)
        pathfiles.append(pathfile)

spi = xr.open_mfdataset(pathfiles, combine='by_coords')

spi30 = spi.spi_30day[29:,:,:]
spi30 = xr.where(spi30 == np.inf, np.nan, spi30) #remove any inf values

spi_time_only_average = spi30.mean(dim=["lat", "lon"], skipna=True).compute()

df = xr.DataArray(
                data=spi_time_only_average,
                name='time_avg'
                )

df.to_netcdf('/scratch/bpuxley/spi_time_average.nc')
'''

#########################################################################################
#Import Data - load previously made time averaged file
#########################################################################################
dirname= '/data2/bpuxley/SPI_30day/spi_time_average.nc'
spi = xr.open_dataset(dirname)

spi30 = spi.time_avg.load()

#########################################################################################
#Calculate the autocorrelation of SPI at each individual grid point
#########################################################################################
lag = 80 #in days
corr = functions.pearsons_corr(spi30, spi30, 0, lag, 'greater')

#########################################################################################
#Calculate the significance
#########################################################################################
alpha1 = 0.05

stipple = []
for i in tqdm(range(0,lag+1)):
	if corr.pvalue[i] <= alpha1:
		stipple.append('True')
	else:
		stipple.append('False')

#########################################################################################
#Plot the correlation
#########################################################################################
fig = plt.figure(figsize = (10,10), dpi = 300, tight_layout =True)

x = np.arange(0,lag+1,1)

# First subplot of the area distributions of drought-to-pluvial (histogram: Frequency vs Area)
ax1 = fig.add_subplot(111)
plt.grid(True, zorder=1)
plt.scatter(x, corr.corr_coef, color='cornflowerblue', zorder=2)
plt.xticks(np.arange(0, lag+1, 2))
#ax1.set_xticklabels(['0','','2', '','4', '', '6', '', '8'], fontsize=10)
plt.yticks(np.arange(0, 1.1, 0.1))
#ax1.set_yticklabels(['0','','5,000','','10,000','','15,000', '','20,000','','25,000','','30,000'], fontsize=10)

for i in range(0, lag+1):
	if stipple[i] == 'True':
		ax1.scatter(x[i], corr.corr_coef[i], color="black", marker='*', s=100, zorder =2) 
	else:
		pass

ax1.set_xlabel('Lag (Days)', fontsize = 10)
ax1.set_ylabel('Correlation', fontsize = 10)
plt.title("a) Lag Autocorrelation of SPI (CONUS averaged)", fontsize=12)

#plt.show(block=False)

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/spi_autocorr.png', bbox_inches = 'tight', pad_inches = 0.1)    

