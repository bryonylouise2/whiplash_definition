#########################################################################################
## A script to create a map of the event climatology across the CONUS
## i.e. total number of times that a grid point is within an extreme-event polygon 
## over the period 1915-2020
## Bryony Louise
## Last Edited: Friday February 26th 2025 
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
from cartopy.io import shapereader
import spei as si
import pandas as pd
import scipy.stats as scs
import shapely.wkt
import os
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import geopandas as gpd

#########################################################################################
#Import Functions
#########################################################################################
import functions

def event_mask(lons, lats, polygon):
	array = np.ones((lats.shape[0], lons.shape[1])) #create an array of ones the size of the lat,lon grid
	mask = functions._mask_outside_region(lons, lats, polygon) #create polygon mask
	masked_array = np.ma.masked_array(array, ~mask)  #mask region outside polygon
	masked_array_filled = np.ma.filled(masked_array, 0) #fill mask with zeros

	return masked_array_filled

#########################################################################################
#Import Events - load previously made event files
#########################################################################################
events_DP = pd.read_csv('/data2/bpuxley/Events/independent_events_DP.csv')
events_PD = pd.read_csv('/data2/bpuxley/Events/independent_events_PD.csv')

df_DP = events_DP.copy()
df_PD = events_PD.copy()

#########################################################################################
#Create a lat, lon grid
#########################################################################################
lat = np.arange(25.15625, 50.03125, 0.0625)
lon = np.arange(235.40625, 293.03125, 0.0625) 
lons, lats = np.meshgrid(lon,lat) #Livneh Grid

o,p = lats.shape

#########################################################################################
#Polygons
#########################################################################################
polygons_DP = [shapely.wkt.loads(i) for i in df_DP.geometry] #convert from nasty string of lat,lons to geometry object
polygons_PD = [shapely.wkt.loads(i) for i in df_PD.geometry] #convert from nasty string of lat,lons to geometry object

#########################################################################################
#Calculate the total number of times that a grid point is within an event polygon
#########################################################################################
#Drought-to-Pluvial
event_freq_DP = np.zeros((lats.shape[0], lons.shape[1]))

for i,(poly) in tqdm(enumerate(polygons_DP)):
	masked_array = event_mask(lons, lats, poly) #create the mask array for each polygon where 1 is inside polygon and 0 is outside polygon
	
	event_freq_DP = np.ma.add(event_freq_DP, masked_array) #add together each masked array
	
#Pluvial-to-Drought
event_freq_PD = np.zeros((lats.shape[0], lons.shape[1]))

for i,(poly) in tqdm(enumerate(polygons_PD)):
	masked_array = event_mask(lons, lats, poly) #create the mask array for each polygon where 1 is inside polygon and 0 is outside polygon
	
	event_freq_PD = np.ma.add(event_freq_PD, masked_array) #add together each masked array
	

#########################################################################################
#Calculate the most common month of an event at each grid point
#########################################################################################
#create a new column in the dataframe that holds the coordinate couples of the polygons

#Drought-to-Pluvial
df_DP['poly_coords'] = np.nan
df_DP['poly_coords'] = df_DP['poly_coords'].astype(object) 

for i in tqdm(range(0,len(polygons_DP))):
	poly = polygons_DP[i]
	coords = list(tuple(poly.exterior.coords))
	df_DP.at[i, 'poly_coords'] = coords

#Pluvial-to-drought
df_PD['poly_coords'] = np.nan
df_PD['poly_coords'] = df_PD['poly_coords'].astype(object) 

for i in tqdm(range(0,len(polygons_PD))):
	poly = polygons_PD[i]
	coords = list(tuple(poly.exterior.coords))
	df_PD.at[i, 'poly_coords'] = coords


#Calculate the most common month at each grid point
most_common_month_DP = np.zeros((o,p))*np.nan
most_common_month_PD = np.zeros((o,p))*np.nan

for i in tqdm(range(0, len(lat))):
	for j in tqdm(range(0, len(lon))):
		latitude = lat[i]
		longitude = lon[j]
		grid_point = (longitude, latitude)
		
		subset_DP = df_DP[df_DP['poly_coords'].apply(lambda x: grid_point in x)]
		subset_PD = df_PD[df_PD['poly_coords'].apply(lambda x: grid_point in x)]
		
		if len(subset_DP>0):
			print('DP len:'+str(len(subset_DP)))
			most_common_month_DP[i,j] = scs.mode(pd.to_datetime(subset_DP.Whiplash_Date).dt.month).mode
		else:
			most_common_month_DP[i,j] = np.nan
			
		if len(subset_PD>0):
			print('PD len:'+str(len(subset_PD)))
			most_common_month_PD[i,j] = scs.mode(pd.to_datetime(subset_PD.Whiplash_Date).dt.month).mode
		else:
			most_common_month_PD[i,j] = np.nan
			
#Save out files so I do not need to rerun
dimensions=['lat','lon']
coords = {
			'lat': lat,
			'lon': lon
			}
DP_xarray = xr.DataArray(most_common_month_DP, coords, dims=dimensions, name='Drought to Pluvial')
PD_xarray = xr.DataArray(most_common_month_PD, coords, dims=dimensions, name='Pluvial to Drought')
dataset = xr.Dataset({"DP_common_month":DP_xarray, "PD_common_month":PD_xarray})

dataset.to_netcdf('/home/bpuxley/most_common_month.nc')
print('file saved')

#########################################################################################
#Create CONUS mask
#########################################################################################
usa = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_20m.zip")
lower_48 = usa[~usa["STUSPS"].isin(["AK", "HI", "PR"])]

lower_48_geom = lower_48.unary_union  
mask = np.zeros(lons.shape, dtype=bool)

lat = np.arange(25.15625, 50.03125, 0.0625)
lon = np.arange(-124.59375, -66.96875, 0.0625)
lons, lats = np.meshgrid(lon,lat) #Livneh Grid

for i in tqdm(range(lons.shape[0])):
    for j in range(lons.shape[1]):
        point = Point(lons[i, j], lats[i, j])
        mask[i, j] = lower_48_geom.contains(point)  # True if inside, False if outside

event_freq_dp_masked = np.where(mask, event_freq_DP, np.nan)  # Set values outside the USA to NaN
event_freq_pd_masked = np.where(mask, event_freq_PD, np.nan)  # Set values outside the USA to NaN


#########################################################################################
#Plot
#########################################################################################
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)
proj = ccrs.PlateCarree()

#Drought-to-Pluvial
ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

#polygon = conus_poly
#poly_x, poly_y = polygon.exterior.xy
	
#plt.plot(poly_x, poly_y, color='red', linewidth=2, transform=ccrs.PlateCarree())

cs = plt.contourf(lons, lats, event_freq_dp_masked, transform=ccrs.PlateCarree(), levels=np.arange(0, 350, 10), cmap = 'YlGnBu') 
#mask.plot()

plt.title("Total Number of Times a Grid Point \nis in an Event Polygon", loc = "left")
plt.title("Drought-to-Pluvial", loc = 'right')

fig.colorbar(cs, ax=ax1, orientation='horizontal', pad=0.05)

#Pluvial-to-Drought
ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linewidth=1)
ax2.add_feature(cfeature.STATES, edgecolor='black')

ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs = plt.contourf(lons, lats, event_freq_pd_masked, transform=ccrs.PlateCarree(), levels=np.arange(0, 350, 10), cmap = 'YlGnBu') 

plt.title("Total Number of Times a Grid Point \nis in an Event Polygon", loc = "left")
plt.title("Pluvial-to-Drought", loc = 'right')

fig.colorbar(cs, ax=ax2, orientation='horizontal', pad=0.05)
	
plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_frequency_CONUS.png', bbox_inches = 'tight', pad_inches = 0.1)    








