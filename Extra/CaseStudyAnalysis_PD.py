#########################################################################################
## Examine and Plot Specific Events - Pluvial to Drought
## Bryony Louise Puxley
## Last Edited: Friday, August 8th, 2025
## Input: Decadal SPI, whiplash occurrences, normalized density, and independent 
## pluvial-to-drought events csv. Choose event number or date.
## Output: gifs of SPI during the drought period, SPI during the pluvial period, SPI change,
## and whiplash occurrences.
#########################################################################################
# Import Required Modules
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
import imageio

#########################################################################################
# Read in Data
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

#Whiplash Points from 3.Whiplash_Identification
dirname= '/data2/bpuxley/Whiplash'

pathfiles = []

for i in years:
	filename = 'whiplashes_%s.nc'%(i)
	pathfile = os.path.join(dirname, filename)
	pathfiles.append(pathfile)

whiplashes = xr.open_mfdataset(pathfiles, combine='by_coords')

#Normalized Density from 5.Spatial Calculation
dirname= '/data2/bpuxley/Density'

pathfiles = []

for i in years:
	filename = 'density_%s.nc'%(i)
	pathfile = os.path.join(dirname, filename)
	pathfiles.append(pathfile)

density = xr.open_mfdataset(pathfiles, combine='by_coords')

print('Read in Data')

#########################################################################################
# Reading in Events
#########################################################################################
#Read in events
events_PD = pd.read_csv('/data2/bpuxley/Events/independent_events_PD.csv')
##########################################################################################
# Choose options
##########################################################################################
#Choose whiplash type
df = events_PD

#Can request event number?
event_num = 101
#Can choose date event starts? 
#doi = "2001-04-10

##########################################################################################
# Subset and Isolate Event
##########################################################################################
#subset based on event_no
subset_ind = np.where((df.Event_No == event_num))[0]
event = df.iloc[subset_ind]

#Get polygons
polygons = [shapely.wkt.loads(i) for i in event.geometry] #convert from nasty string of lat,lons to geometry object

#Create list of dates within event
third_column = df.columns[2]
event_dates = event[third_column]

##########################################################################################
# Plot - Change Information depending on whether looking at DP or PD
##########################################################################################
#Lons, lats
lons, lats = np.meshgrid(whiplashes.lon.values, whiplashes.lat.values) #create a meshgrid of lat,lon values

##########################################################################################
# SPI Pluvial
##########################################################################################
pathfiles = []

for i in tqdm(range(0,len(event_dates))):
	date = event_dates.iloc[i]
	whiplash_date = event.Whiplash_Date.iloc[i]
	
	fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

	ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

	ax1.add_feature(cfeature.COASTLINE)
	ax1.add_feature(cfeature.BORDERS, linewidth=1)
	ax1.add_feature(cfeature.STATES, edgecolor='black')

	ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

	cs = plt.contourf(lons, lats, spi.spi_30day.sel(time=whiplash_date).values, transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-3, 3, 0.1), cmap = 'BrBG') 

	polygon = polygons[i] 
	poly_x, poly_y = polygon.exterior.xy
	
	plt.plot(poly_x, poly_y, color='red', linewidth=2, transform=ccrs.PlateCarree())
	
	plt.title("SPI CONUS Pluvial \nDate: %s"%(date), loc = "left")
	plt.title("Event Day %s"%(i), loc = 'right')

	fig.colorbar(cs, ax=ax1, orientation='horizontal', pad=0.05)
	
	pathfile = "/home/bpuxley/Definition_and_Climatology/Plots/Case_Study/SPI_pluvial/Raw_Fig_%s.png" % i 
	pathfiles.append(pathfile)
	
	plt.savefig(pathfile, bbox_inches="tight")

	#plt.show(block=False)


with imageio.get_writer('/home/bpuxley/Definition_and_Climatology/Plots/Case_Study/Pluvial-to-Drought/%s_spi_pluvial.gif'%(event_num), mode='I') as writer:
    for filename in pathfiles:
        image = imageio.imread(filename)
        writer.append_data(image)

#############################################
# Deleting Things 
#############################################
for filename in set(pathfiles):
    os.remove(filename)
    
##########################################################################################
# SPI Drought
##########################################################################################
pathfiles = []

for i in tqdm(range(0,len(event_dates))):
	date = event_dates.iloc[i]
	drought_date = (pd.to_datetime(event.Whiplash_Date.iloc[i]) + timedelta(days=30)).strftime('%Y-%m-%d')
	
	fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

	ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

	ax1.add_feature(cfeature.COASTLINE)
	ax1.add_feature(cfeature.BORDERS, linewidth=1)
	ax1.add_feature(cfeature.STATES, edgecolor='black')

	ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

	cs = plt.contourf(lons, lats, spi.spi_30day.sel(time=drought_date).values, transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-3, 3, 0.1), cmap = 'BrBG') 

	polygon = polygons[i] 
	poly_x, poly_y = polygon.exterior.xy
	
	plt.plot(poly_x, poly_y, color='red', linewidth=2, transform=ccrs.PlateCarree())
	
	plt.title("SPI CONUS Drought \nDate: %s"%(date), loc = "left")
	plt.title("Event Day %s"%(i), loc = 'right')

	fig.colorbar(cs, ax=ax1, orientation='horizontal', pad=0.05)
	
	pathfile = "/home/bpuxley/Definition_and_Climatology/Plots/Case_Study/SPI_drought/Raw_Fig_%s.png" % i 
	pathfiles.append(pathfile)
	
	plt.savefig(pathfile, bbox_inches="tight")

	#plt.show(block=False)


with imageio.get_writer('/home/bpuxley/Definition_and_Climatology/Plots/Case_Study/Pluvial-to-Drought/%s_spi_drought.gif'%(event_num), mode='I') as writer:
    for filename in pathfiles:
        image = imageio.imread(filename)
        writer.append_data(image)

#############################################
# Deleting Things 
#############################################
for filename in set(pathfiles):
    os.remove(filename)

    
##########################################################################################
# SPI Change
##########################################################################################
pathfiles = []

for i in tqdm(range(0,len(event_dates))):
	date = event_dates.iloc[i]
	pluvial_date = event.Whiplash_Date.iloc[i]
	drought_date = (pd.to_datetime(event.Whiplash_Date.iloc[i]) + timedelta(days=30)).strftime('%Y-%m-%d')
	
	spi_change = spi.spi_30day.sel(time=pluvial_date).values - spi.spi_30day.sel(time=drought_date).values
	
	fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

	ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

	ax1.add_feature(cfeature.COASTLINE)
	ax1.add_feature(cfeature.BORDERS, linewidth=1)
	ax1.add_feature(cfeature.STATES, edgecolor='black')

	ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

	cs = plt.contourf(lons, lats, spi_change, transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-5, 5, 0.1), cmap = 'BrBG') 

	polygon = polygons[i] 
	poly_x, poly_y = polygon.exterior.xy
	
	plt.plot(poly_x, poly_y, color='red', linewidth=2, transform=ccrs.PlateCarree())
	
	plt.title("SPI CONUS Change \nDate: %s"%(date), loc = "left")
	plt.title("Event Day %s"%(i), loc = 'right')

	fig.colorbar(cs, ax=ax1, orientation='horizontal', pad=0.05)
	
	pathfile = "/home/bpuxley/Definition_and_Climatology/Plots/Case_Study/SPI_diff/Raw_Fig_%s.png" % i 
	pathfiles.append(pathfile)
	
	plt.savefig(pathfile, bbox_inches="tight")

	#plt.show(block=False)


with imageio.get_writer('/home/bpuxley/Definition_and_Climatology/Plots/Case_Study/Pluvial-to-Drought/%s_spi_change.gif'%(event_num), mode='I') as writer:
    for filename in pathfiles:
        image = imageio.imread(filename)
        writer.append_data(image)

#############################################
# Deleting Things 
#############################################
for filename in set(pathfiles):
    os.remove(filename)


##########################################################################################
# Whiplash Points
##########################################################################################
pathfiles = []

for i in tqdm(range(0,len(event_dates))):
	date = event_dates.iloc[i]
	whiplash_date = event.Whiplash_Date.iloc[i]
	
	fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

	ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

	ax1.add_feature(cfeature.COASTLINE)
	ax1.add_feature(cfeature.BORDERS, linewidth=1)
	ax1.add_feature(cfeature.STATES, edgecolor='black')

	ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

	# Create a custom color map: purple for True, white for False
	cmap = ListedColormap(["white", "purple"])

	cs = ax1.pcolormesh(lons, lats, whiplashes.PD_whiplashes.sel(time=whiplash_date), cmap=cmap, transform=ccrs.PlateCarree())
	
	polygon = polygons[i] 
	poly_x, poly_y = polygon.exterior.xy
	
	plt.plot(poly_x, poly_y, color='red', linewidth=2, transform=ccrs.PlateCarree())
	
	plt.title("Whiplash Occurrences CONUS \nDate: %s"%(date), loc = "left")
	plt.title("Event Day %s"%(i), loc = 'right')

	fig.colorbar(cs, ax=ax1, orientation='horizontal', pad=0.05)
	
	pathfile = "/home/bpuxley/Definition_and_Climatology/Plots/Case_Study/whiplash_points/Raw_Fig_%s.png" % i 
	pathfiles.append(pathfile)
	
	plt.savefig(pathfile, bbox_inches="tight")

	#plt.show(block=False)

with imageio.get_writer('/home/bpuxley/Definition_and_Climatology/Plots/Case_Study/Pluvial-to-Drought/%s_whiplash_occurrences.gif'%(event_num), mode='I') as writer:
    for filename in pathfiles:
        image = imageio.imread(filename)
        writer.append_data(image)

#############################################
# Deleting Things 
#############################################
for filename in set(pathfiles):
    os.remove(filename)
