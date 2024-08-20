#########################################################################################
## Calculate the Standardised Precipitation Index (SPI) for all grid points across the CONUS between 1915-2020.
## Bryony Louise
## Last Edited: Tuesday August 20th 2024 
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
import spei as si
import pandas as pd
import scipy.stats as scs
import os

#########################################################################################
#Import Data
#########################################################################################
filename = 'prec.1915_2020.nc'
dirname= '/home/bpuxley/'

pathfile = os.path.join(dirname, filename)

df_precip = xr.open_dataset(pathfile)
print('read in data')

#########################################################################################
#Regions For Analysis
#########################################################################################
#Choose Region
Region = "SGP"

region_lon = {"test":[277,277.2],
              "CONUS":[235,295], 
              "WC":[235,241],
              "MW_N":[241,255], 
              "MW_S":[241,255],
              "NGP":[255,265], 
              "SGP":[255,265], 
              "GL_N":[265,279],
              "GL_S":[265,279],
              "NE":[279,295],
              "SE_W":[265,275], 
              "SE_E":[275,286]}

region_lat = {"test":[35.1,35.3], 
              "CONUS":[25,50],
              "WC":[30,50],
              "MW_N":[40,50], 
              "MW_S":[25,40],              
              "NGP":[40,50],
              "SGP":[25,40], 
              "GL_N":[43,50], 
              "GL_S":[36,43],
              "NE":[36,50],
              "SE_W":[25,36],
              "SE_E":[25,36]}

inputlon = region_lon[Region]
inputlat = region_lat[Region]

#slice the data to region of interest
prec_obs = df_precip.sel(lat=slice(inputlat[0], inputlat[1]), lon=slice(inputlon[0], inputlon[1]))
m,o,p = len(prec_obs.time), len(prec_obs.lat), len(prec_obs.lon)
del(df_precip)

print('Time: '+ str(m))
print('Lat: '+ str(o))
print('Lon: '+ str(p))

####################################################################################
#Calculate 30-day rolling sum of precipitation throughout the time period
#########################################################################################
print('About to calculate 30-day rolling sum')
window = 30
prec_rolling = prec_obs.rolling(time=window).sum()
print('Calculated 30-day rolling sum')
#########################################################################################
#Calculate Climate Indices: SPI & SPEI
#########################################################################################
#First regrid precipitation data to space,time (2D)
prec_2D = prec_rolling.stack(point=('lat', 'lon'))
print('regridded precipitation data to 2D array')

#create an empty array to store the spi data
spi30_gamma = np.zeros(([m,o*p]))*np.nan

#Loop through each grid point and calculate the SPI
print('starting loop to calculate SPI')
for i in tqdm(range(0, len(prec_2D.point))):
        print(i)
        series = prec_2D.prec[:,i].to_pandas()
        #print(series)
        if series.isnull().all():
                print('is all NaNs')
                spi30_gamma[:,i] = series
        elif np.nanmax(series) == 0.0:
                print('erroneous data')
                spi30_gamma[:,i] = series*np.nan
        else:
                spi30_gamma[:,i] = si.spi(series, dist=scs.gamma, fit_freq="ME")
        #print(spi30_gamma)

#reshape back into a 3D array (time, lat, lon)
spi30_gamma_new = np.reshape(spi30_gamma, [m, o, p], order="F")

#########################################################################################
#Convert the numpy array back to an xarray dataset
#########################################################################################
time = pd.date_range(start='1915-01-01', end='2020-12-31', freq='D')
lats = prec_obs.lat
lons = prec_obs.lon
attrs = prec_obs.attrs


dimensions=['time','lat','lon']
coords = {
                        'time': time,
                        'lat': lats,
                        'lon': lons
                        }
#attributes = {'sources': 'Livneh et al., 2013 & PRISM, 2004', 'references': 'http://www.esrl.noaa.gov/psd/data/gridded/data.livneh.html','}

spi30_xarray = xr.DataArray(spi30_gamma_new, coords, dims=dimensions, name='spi_30day')
spi30_dataset = xr.Dataset({"spi_30day":spi30_xarray})
#########################################################################################
#Save File out as a netCDF file
#########################################################################################
spi30_dataset.to_netcdf('/scratch/bpuxley/SPI30_%s.nc'%(Region))
