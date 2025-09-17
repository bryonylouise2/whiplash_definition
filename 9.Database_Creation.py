########################################################################################
## This script creates the databases of all of the precipitation whiplash events, subsetting 
## by areal threshold (can be set to zero to keep all events) and also calculates relevant 
## statistics such as: the area-averaged SPI during the 30-day drought period, the area-averaged
## SPI during the 30-day pluvial period, the area-averaged SPI change, the grid point magnitude 
## maximum SPI during the 30-day drought and pluvial periods, respectively, and the grid point
## maximum SPI change.
## Bryony Louise Puxley
## Last Edited: Monday, July 28th, 2025 
## Input: Decadal SPI files, decadal whiplash files, decadal density/KDE files, and decadal 
## potential events CSV files.
## Output: Two CSV files of drought-to-pluvial and pluvial-to-drought events.
#########################################################################################
#Import Required Modules
#########################################################################################
import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta
import shapely.wkt

#########################################################################################
#Import Functions
#########################################################################################
import functions

#########################################################################################
#Set Area Threshold
#########################################################################################
AREA = 175000

#########################################################################################
#Import Data - load previously made SPI, whiplash identification, and potential events files
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

#Whiplash Points from 4.Whiplash_Identification
dirname= '/data2/bpuxley/Whiplash'

pathfiles = []

for i in years:
	filename = 'whiplashes_%s.nc'%(i)
	pathfile = os.path.join(dirname, filename)
	pathfiles.append(pathfile)

whiplashes = xr.open_mfdataset(pathfiles, combine='by_coords')

#Normalized Density from 6.Spatial Calculation
dirname= '/data2/bpuxley/Density'

pathfiles = []

for i in years:
	filename = 'density_%s.nc'%(i)
	pathfile = os.path.join(dirname, filename)
	pathfiles.append(pathfile)

density = xr.open_mfdataset(pathfiles, combine='by_coords')

#Potential Events Databases from 8.Area_Calculation
dirname= '/data2/bpuxley/Databases/'

DP_pathfiles = []
PD_pathfiles = []

for i in years:
	DP_filename = 'potential_events_DP_%s.csv'%(i)
	DP_pathfile = os.path.join(dirname, DP_filename)
	DP_pathfiles.append(DP_pathfile)
	
	PD_filename = 'potential_events_PD_%s.csv'%(i)
	PD_pathfile = os.path.join(dirname, PD_filename)
	PD_pathfiles.append(PD_pathfile)
	

potential_events_DP = pd.concat([pd.read_csv(f) for f in DP_pathfiles ], ignore_index=True)
potential_events_PD = pd.concat([pd.read_csv(f) for f in PD_pathfiles ], ignore_index=True)

#Lons, lats
lons, lats = np.meshgrid(whiplashes.lon.values, whiplashes.lat.values) #create a meshgrid of lat,lon values

#########################################################################################
#Subset potential events based on area threshold
#########################################################################################
potential_events_DP = potential_events_DP[potential_events_DP['Area'] >= AREA].reset_index(drop=True)
potential_events_PD = potential_events_PD[potential_events_PD['Area'] >= AREA].reset_index(drop=True)

polygons_DP = [shapely.wkt.loads(i) for i in potential_events_DP.geometry] #convert from nasty string of lat,lons to geometry object
polygons_PD = [shapely.wkt.loads(i) for i in potential_events_PD.geometry] #convert from nasty string of lat,lons to geometry object


#########################################################################################
#Notes for Dates
#########################################################################################
# consider a drought to pluvial whiplash where Jan 1 to Jan 30 is in drought and Jan 31 to March 1 is pluvial
# SPI is saved so that January 30th refers to the 30-day SPI between January 1st and January 30th, etc.
# drought spi: Jan 30
# pluvial spi: Mar 1
# whiplash and density data are saved as the date the change occurred
# here whiplash/density date: Jan 30
# potential events are saved, so drought date is the start of the drought month, and pluvial is the start of the pluvial month
# here, the drought date would be Jan 1, and the pluvial date would be Jan 31

#########################################################################################
#Calculate Relevant Statistics
#########################################################################################
#setup arrays for precip statistics
#Drought-to-Pluvial
spi_drought_avg_DP = np.zeros((len(potential_events_DP.Drought_Date)))
spi_pluvial_avg_DP = np.zeros_like(spi_drought_avg_DP)
spi_change_avg_DP = np.zeros_like(spi_drought_avg_DP)

spi_drought_max_DP = np.zeros_like(spi_drought_avg_DP)
spi_pluvial_max_DP = np.zeros_like(spi_drought_avg_DP)
spi_change_max_DP = np.zeros_like(spi_drought_avg_DP)

print('Drought-to-Pluvial loop: started')
for i,(event_date,poly) in enumerate(zip(potential_events_DP.Drought_Date, polygons_DP)):
	event_date = event_date
	drought_date = potential_events_DP.Whiplash_Date[i]
	pluvial_date = (pd.to_datetime(potential_events_DP.Whiplash_Date[i]) + timedelta(days=30)).strftime('%Y-%m-%d')
	
	spi_drought = spi.spi_30day.sel(time=drought_date)
	spi_pluvial = spi.spi_30day.sel(time=pluvial_date)
	
	stats = functions.calc_polygon_statistics(lons, lats, polygon=poly, spi_drought=spi_drought.values, spi_pluvial=spi_pluvial.values)
	
	spi_drought_avg_DP[i] = stats['drought_spi_area_avg']
	spi_pluvial_avg_DP[i] = stats['pluvial_spi_area_avg']
	spi_change_avg_DP[i] = stats['spi_change_area_avg']
	spi_drought_max_DP[i] = stats['drought_spi_max']
	spi_pluvial_max_DP[i] = stats['pluvial_spi_max']
	spi_change_max_DP[i] = stats['spi_change_max']
print('Drought-to-Pluvial loop: ended')

potential_events_DP['Avg_SPI_Drought'] = spi_drought_avg_DP
potential_events_DP['Avg_SPI_Pluvial'] = spi_pluvial_avg_DP
potential_events_DP['Avg_SPI_Change'] = spi_change_avg_DP
potential_events_DP['MaxMag_SPI_Drought'] = spi_drought_max_DP
potential_events_DP['MaxMag_SPI_Pluvial'] = spi_pluvial_max_DP
potential_events_DP['Max_SPI_Change'] = spi_change_max_DP

print('Drought-to-Pluvial: saving file')
potential_events_DP.to_csv(f'/data2/bpuxley/Events/events_DP.csv', index=False)
print('Drought-to-Pluvial: file saved')

#Pluvial-to-Drought
spi_drought_avg_PD = np.zeros((len(potential_events_PD.Pluvial_Date)))
spi_pluvial_avg_PD = np.zeros_like(spi_drought_avg_PD)
spi_change_avg_PD = np.zeros_like(spi_drought_avg_PD)

spi_drought_max_PD = np.zeros_like(spi_drought_avg_PD)
spi_pluvial_max_PD = np.zeros_like(spi_drought_avg_PD)
spi_change_max_PD = np.zeros_like(spi_drought_avg_PD)

print('Pluvial-to-Drought loop: started')
for i,(event_date,poly) in enumerate(zip(potential_events_PD.Pluvial_Date, polygons_PD)):
	event_date = event_date
	pluvial_date = potential_events_PD.Whiplash_Date[i]
	drought_date = (pd.to_datetime(potential_events_PD.Whiplash_Date[i]) + timedelta(days=30)).strftime('%Y-%m-%d')
	
	spi_drought = spi.spi_30day.sel(time=drought_date)
	spi_pluvial = spi.spi_30day.sel(time=pluvial_date)
	
	stats = functions.calc_polygon_statistics(lons, lats, polygon=poly, spi_drought=spi_drought.values, spi_pluvial=spi_pluvial.values)
	
	spi_drought_avg_PD[i] = stats['drought_spi_area_avg']
	spi_pluvial_avg_PD[i] = stats['pluvial_spi_area_avg']
	spi_change_avg_PD[i] = stats['spi_change_area_avg']
	spi_drought_max_PD[i] = stats['drought_spi_max']
	spi_pluvial_max_PD[i] = stats['pluvial_spi_max']
	spi_change_max_PD[i] = stats['spi_change_max']
print('Pluvial-to-Drought loop: ended')

potential_events_PD['Avg_SPI_Drought'] = spi_drought_avg_PD
potential_events_PD['Avg_SPI_Pluvial'] = spi_pluvial_avg_PD
potential_events_PD['Avg_SPI_Change'] = spi_change_avg_PD
potential_events_PD['MaxMag_SPI_Drought'] = spi_drought_max_PD
potential_events_PD['MaxMag_SPI_Pluvial'] = spi_pluvial_max_PD
potential_events_PD['Max_SPI_Change'] = spi_change_max_PD

ind = np.where((potential_events_PD.Avg_SPI_Drought.isnull()))[0]
nan_rows = potential_events_PD.iloc[ind]
print(nan_rows)
potential_events_PD.drop(ind, inplace=True)
potential_events_PD.reset_index(drop=True)

print('Pluvial-to-Drought: saving file')
potential_events_PD.to_csv(f'/data2/bpuxley/Events/events_PD.csv', index=False)
print('Pluvial-to-Drought:file saved')


 
