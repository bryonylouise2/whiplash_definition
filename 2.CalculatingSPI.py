#########################################################################################
## Calculate the Standardised Precipitation Index (SPI) for all grid points across the 
## CONUS from 1915 to 2020.
## Bryony Louise Puxley
## Last Edited: Friday, July 25th, 2025 
## Input: Precipitation Data
## Output: netCDF file of rolling SPI values (choose from 30-, 60-, 90-, or 180- days) at 
## a ~ 6 km grid resolution from 1915 to 2020; and a PNG file of the average SPI across 
## the chosen region for visualization.
#########################################################################################
# Import Required Modules
#########################################################################################
import os
import numpy as np
import pandas as pd
import scipy.stats as scs
import xarray as xr
from tqdm import tqdm

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#########################################################################################
# Import Data
#########################################################################################
filename = 'prec.1915_2020.nc'
dirname= '/home/bpuxley/'

pathfile = os.path.join(dirname, filename)

df_precip = xr.open_dataset(pathfile)
print('read in data')

#########################################################################################
# Regions For Analysis (Split into regions to save on memory or run on entire CONUS)
#########################################################################################
#Choose Region
Region = "WCN"

region_lon = {"test":[277,277.2],
              "CONUS":[235,295], 
              "WCN":[235,241], #west coast north
              "WCS":[235,241], #west coast south
              "MWN":[241,255], #mountain west north
              "MWC":[241,255], #moutain west central
              "MWS":[241,255], #mountain west south
              "NGP":[255,265], #northern great plains
              "SGP":[255,265], #southern great plains
              "NGL":[265,279], #northern great lakes
              "SGL":[265,279], #southern great lakes
              "NNE":[279,295], #northern north east
              "SNE":[279,295], #southern north east
              "WSE":[265,279], #western south east
              "ESE":[279,295]} #eastern south east

region_lat = {"test":[35.1,35.3], 
              "CONUS":[25,50],
              "WCN":[40,50], #west coast north
              "WCS":[25,40], #west coast south
              "MWN":[42,50], #mountain west north 
              "MWC":[33,42], #moutain west central
              "MWS":[25,33], #moutain west south              
              "NGP":[40,50], #northern great plains
              "SGP":[25,40], #southern great plains
              "NGL":[43,50], #northern great lakes
              "SGL":[36,43], #southern great lakes
              "SNE":[36,43], #southern north east
              "NNE":[43,50], #northern north east
              "WSE":[25,36], #western south east 
              "ESE":[25,36]} #eastern south east

inputlon = region_lon[Region]
inputlat = region_lat[Region]

#slice the data to the region of interest
prec_obs = df_precip.sel(lat=slice(inputlat[0], inputlat[1]), lon=slice(inputlon[0], inputlon[1]))
m,o,p = len(prec_obs.time), len(prec_obs.lat), len(prec_obs.lon)
del(df_precip)

print('Time: '+ str(m))
print('Lat: '+ str(o))
print('Lon: '+ str(p))

####################################################################################
# Calculate the  rolling sum of precipitation throughout the period 
# Chose between 30-, 60-, 90-, or 180- day rolling sums by changing the window
#########################################################################################
print('About to calculate 30-day rolling sum')
window = 30
prec_rolling = prec_obs.rolling(time=window).sum()
print('Calculated 30-day rolling sum')

#########################################################################################
# Calculate Climate Indices: SPI
#########################################################################################
#First regrid precipitation data to space, time (2D)
prec_2D = prec_rolling.stack(point=('lat', 'lon'))
print('regridded precipitation data to 2D array')

#create an empty array to store the spi data
spi_gamma = np.zeros(([m,o*p]))*np.nan

#Loop through each grid point and calculate the SPI
print('starting loop to calculate SPI')
for i in tqdm(range(0, len(prec_2D.point))):
        print(i)
        series = prec_2D.prec[:,i].to_pandas()
        #print(series)
        if series.isnull().all(): #if the whole series is nan, make spi values all nans
                print('is all NaNs')
                spi_gamma[:,i] = series
        elif series[0:round(len(series)*3/4)].isnull().all(): #if 3/4 of the whole series is nan, make spi values all nans
                print('is 3/4 NaNs')
                spi_gamma[:,i] = series*np.nan
        elif np.nanmax(series) == 0.0: #if the data is all zeros, make spi values all nans
                print('erroneous data')
                spi_gamma[:,i] = series*np.nan
        else: #else calculate the spi
                spi_gamma[:,i] = si.spi(series, dist=scs.gamma, fit_freq="ME")
        #print(spi_gamma)

#reshape back into a 3D array (time, lat, lon)
spi_gamma_new = np.reshape(spi_gamma, [m, o, p]) #This line may be incorrect; may need to specify reshape method or unstack first. Check the plot after computing.

#########################################################################################
# Convert the numpy array back to an xarray dataset
#########################################################################################
time = pd.date_range(start='1915-01-01', end='2020-12-31', freq='D')
lats = prec_obs.lat
lons = prec_obs.lon
attrs = prec_obs.attrs

dimensions=['time', 'lat', 'lon']
coords = {
                        'time': time,
                        'lat': lats,
                        'lon': lons
                        }
#attributes = {'sources': 'Livneh et al., 2013 & PRISM, 2004', 'references': 'http://www.esrl.noaa.gov/psd/data/gridded/data.livneh.html','}

spi_xarray = xr.DataArray(spi_gamma_new, coords, dims=dimensions, name='spi_%s_day'%(window))
spi_dataset = xr.Dataset({'spi_%s_day'%(window):spi_xarray})
#########################################################################################
# Save File out as a netCDF file
#########################################################################################
spi_dataset.to_netcdf('/scratch/bpuxley/SPI_%s_%s.nc'%(window, Region))

#########################################################################################
# Calculate and Plot the Average SPI for the region - to check if it looks correct
#########################################################################################
#Calculate the average SPI over the time period at each grid point
spi_sum = np.nansum(spi_gamma_new, axis=0)
spi_mean = spi_sum/m

lon, lat = np.meshgrid(prec_obs.lon.values, prec_obs.lat.values)
#########################################################################################
# Plot the SPI data
#########################################################################################
fig = plt.figure(figsize = (6,7), dpi = 300, tight_layout =True)

# Second subplot with reshaped spi
ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs1 = plt.contourf(lon, lat, spi_mean, transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(-0.1, 0.1, 0.01), cmap = 'BrBG')

fig.colorbar(cs1, ax=ax1, orientation='horizontal', pad=0.05)

plt.title('SPI Mean %s'%(Region))

plt.savefig('/scratch/bpuxley/Plots/spi_%s.png'%(Region), bbox_inches = 'tight', pad_inches = 0.1)    
