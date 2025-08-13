#########################################################################################
## Identify individual drought and pluvial occurrences for all grid points across the CONUS 
## from 1915 to 2020.
## Bryony Louise Puxley
## Last Edited: Wednesday, August 13th, 2025 
## Input: Regional SPI files or CONUS-wide SPI file - need all times at each grid point.
## Output: netCDF file of identified drought and pluvial occurrences at a ~ 6 km grid 
## resolution from 1915 to 2020.
#########################################################################################
# Import Required Modules
#########################################################################################
import xesmf as xe
import numpy as np
import xarray as xr
from tqdm import tqdm
import time
from datetime import datetime, timedelta, date
from netCDF4 import Dataset, num2date, MFDataset
import pandas as pd
import scipy.stats as scs
import os
import gzip

#########################################################################################
# Import Functions
#########################################################################################
import functions 

#########################################################################################
#Identify occurrences of drought and pluvial events
#########################################################################################
regions = {"WCN", "WCS", "MWN", "MWC", "MWS", "NGP", "SGP", "NGL", "SGL", "NNE", "SNE", "ESE", "WSE"}

regions_info = {"WCN": {'lon':[235,241], 'lat':[40,50]}, #west coast north
				"WCS": {'lon':[235,241], 'lat':[25,40]}, #west coast south
				"MWN": {'lon':[241,255], 'lat':[42,50]}, #mountain west north
				"MWC": {'lon':[241,255], 'lat':[33,42]}, #mountain west central
				"MWS": {'lon':[241,255], 'lat':[25,33]}, #mountain west south
				"NGP": {'lon':[255,265], 'lat':[40,50]}, #northern great plains
				"SGP": {'lon':[255,265], 'lat':[25,40]}, #southern great plains
				"NGL": {'lon':[265,279], 'lat':[43,50]}, #northern great lakes
				"SGL": {'lon':[265,279], 'lat':[36,43]}, #southern great lakes
				"NNE": {'lon':[279,295], 'lat':[36,43]}, #northern north east
				"SNE": {'lon':[279,295], 'lat':[43,50]}, #southern north east
				"WSE": {'lon':[265,275], 'lat':[25,36]}, #western south east
				"ESE": {'lon':[275,286], 'lat':[25,36]}} #eastern south east
			
for region in sorted(regions):
	print(region)
	
	################################################################################################
	# import data
	################################################################################################
	window = 30 #choose from 60-, 90-, or 180
	filename = 'SPI_%s_%s.nc'%(window, region)
	dirname= '/data2/bpuxley/SPI_30day/regional_data'

	pathfile = os.path.join(dirname, filename)

	df_spi = xr.open_dataset(pathfile)
	print('Read in Data')

	m,o,p = len(df_spi.time), len(df_spi.lat), len(df_spi.lon) #length of data #m,o,p: time, lat, lon 
	lon, lat = np.meshgrid(df_spi.lon.values, df_spi.lat.values) #create a meshgrid of lat,lon values

	print('Time: '+ str(m))
	print('Lat: '+ str(o))
	print('Lon: '+ str(p))

	'''
	################################################################################################
	# Loop through each grid point and identify occurrences of ALL droughts and pluvials
	################################################################################################
	# Droughts are when the 30-day SPI is less than -1 and Pluvials are when the 30-day SPI 
	# is greater than +1
	################################################################################################
	spi = df_spi.spi_30_day #create variable spi which contains the spi data from the data array
	del(df_spi)

	binary_array_drought = np.zeros((m,o,p))*np.nan #create an array to store the binary array of drought occurrences
	binary_array_pluvial = np.zeros((m,o,p))*np.nan #create an array to store the binary array of pluvial occurrences

	#Loop through all days and identify the grid points that experience events 
	print('Starting loop to identify events')
	for i in tqdm(range(window-1, m-window)):
	#i.e, for rolling 30-day periods; 
	## loop through the timeseries from value 29 (first 30 values will be all nans) up until the last 30 days
	#Look for DP events by comparing the SPI value at day i to the SPI value at day i+30
	#day i is the 30 day SPI from day i-30 to day i; day i+30 is the 30 day SPI from day i to day i+30
	#If a window of 60-,90-, or 180- days was chosen, just replace 30 with that value
	
		#drought events
		bool_array = xr.where((spi[i].values <= -1),True,False) #find all the grid points that experience a drought event
		binary_array_drought[i,:,:] = bool_array
	
		#pluvial events
		bool_array = xr.where((spi[i].values >= +1),True,False) #find all the grid points that experience a pluvial event
		binary_array_pluvial[i,:,:] = bool_array

	print('events identifited')
	
	'''
	################################################################################################
	# OR Loop through each grid point and identify occurrences of droughts and pluvials with either
	# dry/wet or normal conditions prior i.e. no whiplash events.
	################################################################################################
	# Droughts are when the 30-day SPI is less than -1 and Pluvials are when the 30-day SPI 
	# is greater than +1
	################################################################################################
	spi = df_spi.spi_30day #create variable spi which contains the spi data from the data array
	del(df_spi)

	binary_array_drought = np.zeros((m,o,p))*np.nan #create an array to store the binary array of drought occurrences
	binary_array_pluvial = np.zeros((m,o,p))*np.nan #create an array to store the binary array of pluvial occurrences

	#Loop through all days and identify the grid points that experience events 
	print('Starting loop to identify events')
	for i in tqdm(range(window-1, m-window)):
	#i.e, for rolling 30-day periods; 
	## loop through the timeseries from value 29 (first 30 values will be all nans) up until the last 30 days
	#Look for DP events by comparing the SPI value at day i to the SPI value at day i+30
	#day i is the 30 day SPI from day i-30 to day i; day i+30 is the 30 day SPI from day i to day i+30
	#If a window of 60-,90-, or 180- days was chosen, just replace 30 with that value
	
		#drought events
		bool_array = xr.where((spi[i+window].values <= -1) & (spi[i].values < +1),True,False) #find all the grid points that experience a drought event that was either dry or normal before. No pluvial-to-drought.
		binary_array_drought[i,:,:] = bool_array
		
		#pluvial events
		bool_array = xr.where((spi[i+window].values >= +1) & (spi[i].values > -1),True,False) #find all the grid points that experience a pluvial event that was either wet or normal before. No drought-to-pluvial.
		binary_array_pluvial[i,:,:] = bool_array

	print('events identifited')

	#########################################################################################
	#Save out the binary file of grid points meeting criteria
	#########################################################################################
	time = pd.date_range(start='1915-01-01', end='2020-12-31', freq='D')
	lats = spi.lat
	lons = spi.lon

	dimensions=['time','lat','lon']
	coords = {
			'time': time,
			'lat': lats,
			'lon': lons
			}

	drought_xarray = xr.DataArray(binary_array_drought, coords, dims=dimensions, name='Drought Events')
	pluvial_xarray = xr.DataArray(binary_array_pluvial, coords, dims=dimensions, name='Pluvial Events')
	dataset = xr.Dataset({"drought_events":drought_xarray, "pluvial_events":pluvial_xarray})

	dataset.to_netcdf('/data2/bpuxley/droughts_and_pluvials/regional_data/droughts_and_pluvials_%s.nc'%(region))
	print('binary file saved')
	
	del(spi)
	del(binary_array_drought)
	del(binary_array_pluvial)
	del(drought_xarray)
	del(pluvial_xarray)
	del(dataset)
	
	print('Region: '+region+' completed')
