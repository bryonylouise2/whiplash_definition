#########################################################################################
## Create the databases of precipitation whiplash events
## Bryony Louise Puxley
## Last Edited: Wedne February 6th 2025 
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
#Set Area Threshhold
#########################################################################################
AREA = 0

#########################################################################################
#Import Data - load previously made spi, whiplash identification and potential events files
#########################################################################################
years = ['1915_1924', '1925_1934', '1935_1944', '1945_1954', '1955_1964', '1965_1974', 
			'1975_1984', '1985_1994', '1995_2004', '2005_2014', '2015_2020']

#SPI from 2.Calculating_SPI
dirname= '/data2/bpuxley/SPI_30day'

pathfiles = []

for i in years:
	filename = 'spi_%s.nc'%(i)
	pathfile = os.path.join(dirname, filename)
	pathfiles.append(pathfile)

spi = xr.open_mfdataset(pathfiles, combine='by_coords')

#Normalized Density from 5.Spatial Calculation
dirname= '/data2/bpuxley/droughts_and_pluvials/density'

droughts_pathfiles = []
pluvials_pathfiles = []

for i in years:
	droughts_filename = 'density_droughts_%s.nc'%(i)
	droughts_pathfile = os.path.join(dirname, droughts_filename)
	droughts_pathfiles.append(droughts_pathfile)

	pluvials_filename = 'density_pluvials_%s.nc'%(i)
	pluvials_pathfile = os.path.join(dirname, pluvials_filename)
	pluvials_pathfiles.append(pluvials_pathfile)

drought_density = xr.open_mfdataset(droughts_pathfiles, combine='by_coords')
pluvial_density = xr.open_mfdataset(pluvials_pathfiles, combine='by_coords')

#Potential Events Databases from 7.Area_Calculation
dirname= '/data2/bpuxley/droughts_and_pluvials/Databases/'

droughts_pathfiles = []
pluvials_pathfiles = []

for i in years:
	droughts_filename = 'potential_events_droughts_%s.csv'%(i)
	droughts_pathfile = os.path.join(dirname, droughts_filename)
	droughts_pathfiles.append(droughts_pathfile)
	
	pluvials_filename = 'potential_events_pluvials_%s.csv'%(i)
	pluvials_pathfile = os.path.join(dirname, pluvials_filename)
	pluvials_pathfiles.append(pluvials_pathfile)
	

potential_events_droughts = pd.concat([pd.read_csv(f) for f in droughts_pathfiles ], ignore_index=True)
potential_events_pluvials = pd.concat([pd.read_csv(f) for f in pluvials_pathfiles ], ignore_index=True)

#Lons, lats
lons, lats = np.meshgrid(spi.lon.values, spi.lat.values) #create a meshgrid of lat,lon values

#########################################################################################
#Subset potential events based on area threshold
#########################################################################################
potential_events_droughts = potential_events_droughts[potential_events_droughts['Area'] >= AREA].reset_index(drop=True)
potential_events_pluvials = potential_events_pluvials[potential_events_pluvials['Area'] >= AREA].reset_index(drop=True)

polygons_droughts = [shapely.wkt.loads(i) for i in potential_events_droughts.geometry] #convert from nasty string of lat,lons to geometry object
polygons_pluvials = [shapely.wkt.loads(i) for i in potential_events_pluvials.geometry] #convert from nasty string of lat,lons to geometry object


#########################################################################################
#Calculate Relevant Statistics
#########################################################################################
#########################################################################################
#Split into smaller chunks for processing
#########################################################################################
#########################################################################################
#Droughts
#########################################################################################
drought_chunk = potential_events_droughts[2000:3000]
polygons_droughts_chunk = polygons_droughts[2000:3000]

spi_drought_avg = np.zeros((len(drought_chunk)))
spi_drought_max = np.zeros_like(spi_drought_avg)

for i,(event_date,poly) in enumerate(tqdm(zip(drought_chunk.Drought_Date, polygons_droughts_chunk))):
	event_date = event_date
	drought_date = pd.DatetimeIndex([pd.to_datetime(event_date) + pd.DateOffset(days=29)])
	
	spi_drought = spi.spi_30day.sel(time=drought_date)
	
	stats = functions.calc_polygon_statistics_drought_pluvial_only(lons, lats, spi=spi_drought.values, polygon=poly, max_process = 'drought')
	
	spi_drought_avg[i] = stats['spi_area_avg']
	spi_drought_max[i] = stats['spi_max']
		
drought_chunk['Avg_SPI_Drought'] = spi_drought_avg
drought_chunk['MaxMag_SPI_Drought'] = spi_drought_max
	
drought_chunk.to_csv(f'/data2/bpuxley/droughts_and_pluvials/Events/events_droughts_%s_2000_2999.csv'%(AREA), index=False)
print('saved Drought events')
 
#########################################################################################
#Pluvials
#########################################################################################
pluvial_chunk = potential_events_pluvials[7000:]
polygons_pluvials_chunk = polygons_pluvials[7000:]

spi_pluvial_avg = np.zeros((len(pluvial_chunk)))
spi_pluvial_max = np.zeros_like(spi_pluvial_avg)

for i,(event_date,poly) in tqdm(enumerate(zip(pluvial_chunk.Pluvial_Date, polygons_pluvials_chunk))):
	event_date = event_date
	pluvial_date = pd.DatetimeIndex([pd.to_datetime(event_date) + pd.DateOffset(days=29)])
	
	spi_pluvial = spi.spi_30day.sel(time=pluvial_date)
	
	stats = functions.calc_polygon_statistics_drought_pluvial_only(lons, lats, spi=spi_pluvial.values, polygon=poly, max_process = 'pluvial')
	
	spi_pluvial_avg[i] = stats['spi_area_avg']
	spi_pluvial_max[i] = stats['spi_max']
		
pluvial_chunk['Avg_SPI_Pluvial'] = spi_pluvial_avg
pluvial_chunk['MaxMag_SPI_Pluvial'] = spi_pluvial_max
	
pluvial_chunk.to_csv(f'/data2/bpuxley/droughts_and_pluvials/Events/events_pluvials_%s_1000_1999.csv'%(AREA), index=False)
print('saved Pluvial events')

#########################################################################################
#Read in all chunks, combine into 1 csv each and save
#########################################################################################
filenames = ['0_1999','2000_3999','4000_4999','5000_5999','6000_6999','7000_end']
dirname= '/data2/bpuxley/droughts_and_pluvials/Events/'

droughts_pathfiles = []
pluvials_pathfiles = []

for i in filenames:
	droughts_filename = 'events_droughts_%s.csv'%(i)
	droughts_pathfile = os.path.join(dirname, droughts_filename)
	droughts_pathfiles.append(droughts_pathfile)
	
	pluvials_filename = 'events_pluvials_%s.csv'%(i)
	pluvials_pathfile = os.path.join(dirname, pluvials_filename)
	pluvials_pathfiles.append(pluvials_pathfile)
	

events_droughts = pd.concat([pd.read_csv(f) for f in droughts_pathfiles ], ignore_index=True)
events_pluvials = pd.concat([pd.read_csv(f) for f in pluvials_pathfiles ], ignore_index=True)

events_droughts.to_csv(f'/data2/bpuxley/droughts_and_pluvials/Events/events_droughts.csv', index=False)
events_pluvials.to_csv(f'/data2/bpuxley/droughts_and_pluvials/Events/events_pluvials.csv', index=False)
print('saved events')


 
