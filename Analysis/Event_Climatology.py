#########################################################################################
## A script to create a map of the event climatology across the CONUS
## i.e., total number of times and the most common month that a grid point is within an 
## extreme-event polygon over the period 1915-2020
## Bryony Louise Puxley
## Last Edited: Monday, August 11th, 2025 
## Input: Independent event files of Drought-to-Pluvial, Pluvial-to-Drought, Drought,
## and Pluvial events.
## Output: Two PNG files. Figure 5 from the journal article: grid point frequency for the
## period 1915-2020 for a) drought-to-pluvial, b) pluvial-to-drought, c) drought, and d) 
## pluvial events. Figure 6 from the journal article: The most common season that a grid point 
## experiences a) drought-to-pluvial, b) pluvial-to-drought, c) drought, and d) pluvial events.
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
# Import Functions
#########################################################################################
import functions

def event_mask(lons, lats, polygon):
	array = np.ones((lats.shape[0], lons.shape[1])) #create an array of ones the size of the lat,lon grid
	mask = functions._mask_outside_region(lons, lats, polygon) #create polygon mask
	masked_array = np.ma.masked_array(array, ~mask)  #mask region outside polygon
	masked_array_filled = np.ma.filled(masked_array, 0) #fill mask with zeros

	return masked_array_filled

#########################################################################################
# Import Events - load previously made event files
#########################################################################################
events_DP = pd.read_csv('/data2/bpuxley/Events/independent_events_DP.csv')
events_PD = pd.read_csv('/data2/bpuxley/Events/independent_events_PD.csv')
events_droughts = pd.read_csv('/data2/bpuxley/droughts_and_pluvials/Events/independent_events_droughts.csv')
events_pluvials = pd.read_csv('/data2/bpuxley/droughts_and_pluvials/Events/independent_events_pluvials.csv')

df_DP =  events_DP.iloc[np.where((events_DP.Day_No == 0))].reset_index(drop=True) 
df_PD =  events_PD.iloc[np.where((events_PD.Day_No == 0))].reset_index(drop=True) 
df_droughts =  events_droughts.iloc[np.where((events_droughts.Day_No == 0))].reset_index(drop=True) 
df_pluvials =  events_pluvials.iloc[np.where((events_pluvials.Day_No == 0))].reset_index(drop=True) 

no_of_events_dp = np.nanmax(events_DP.Event_No)
no_of_events_pd = np.nanmax(events_PD.Event_No)
no_of_events_droughts = np.nanmax(events_droughts.Event_No)
no_of_events_pluvials = np.nanmax(events_pluvials.Event_No)

years = np.arange(1915,2021,1)

#########################################################################################
# Create a lat, lon grid
#########################################################################################
lat = np.arange(25.15625, 50.03125, 0.0625)
lon = np.arange(235.40625, 293.03125, 0.0625) 
lons, lats = np.meshgrid(lon,lat) #Livneh Grid

o,p = lats.shape

#########################################################################################
# Polygons
#########################################################################################
polygons_DP = [shapely.wkt.loads(i) for i in df_DP.geometry] #convert from nasty string of lat,lons to geometry object
polygons_PD = [shapely.wkt.loads(i) for i in df_PD.geometry] #convert from nasty string of lat,lons to geometry object
polygons_droughts = [shapely.wkt.loads(i) for i in df_droughts.geometry] #convert from nasty string of lat,lons to geometry object
polygons_pluvials = [shapely.wkt.loads(i) for i in df_pluvials.geometry] #convert from nasty string of lat,lons to geometry object

df_DP['polygon'] = polygons_DP
df_PD['polygon'] = polygons_PD
df_droughts['polygon'] = polygons_droughts
df_pluvials['polygon'] = polygons_pluvials

#########################################################################################
# Calculate the total number of times that a grid point is within an event polygon
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
	
#Droughts
event_freq_droughts = np.zeros((lats.shape[0], lons.shape[1]))

for i,(poly) in tqdm(enumerate(polygons_droughts)):
	masked_array = event_mask(lons, lats, poly) #create the mask array for each polygon where 1 is inside polygon and 0 is outside polygon
	
	event_freq_droughts = np.ma.add(event_freq_droughts, masked_array) #add together each masked array
	
#Pluvials
event_freq_pluvials = np.zeros((lats.shape[0], lons.shape[1]))

for i,(poly) in tqdm(enumerate(polygons_pluvials)):
	masked_array = event_mask(lons, lats, poly) #create the mask array for each polygon where 1 is inside polygon and 0 is outside polygon
	
	event_freq_pluvials = np.ma.add(event_freq_pluvials, masked_array) #add together each masked array

#########################################################################################
# Calculate the most common month of an event at each grid point - run on oscer
#########################################################################################
'''
#Calculate the most common month at each grid point
most_common_month_DP = np.zeros((o,p))*np.nan
most_common_month_PD = np.zeros((o,p))*np.nan

for i in tqdm(range(0, len(lat))):
	for j in tqdm(range(0, len(lon))):
		latitude = lat[i]
		longitude = lon[j]
		grid_point = (longitude, latitude)
		
		subset_DP = df_DP[df_DP['polygon'].apply(lambda poly: poly.contains(Point(grid_point)))]
		subset_PD = df_PD[df_PD['polygon'].apply(lambda poly: poly.contains(Point(grid_point)))]
			
		if len(subset_DP)>0:
			most_common_month_DP[i,j] = scs.mode(pd.to_datetime(subset_DP.Whiplash_Date).dt.month).mode
		else:
			most_common_month_DP[i,j] = np.nan
			
		if len(subset_PD)>0:
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

dataset.to_netcdf('/data2/bpuxley/Events/most_common_month_alldays.nc')
print('file saved')
'''
#########################################################################################
# Load in file (whiplash events)
#########################################################################################
dirname= '/data2/bpuxley/Events/most_common_month_day0only.nc'
most_common_month = xr.open_dataset(dirname)

most_common_month_dp = most_common_month.DP_common_month
most_common_month_pd = most_common_month.PD_common_month

# Define conditions and corresponding values
choices = [1, 2, 3, 4]
conditions_dp = [
    np.isin(most_common_month_dp, [12, 1, 2]),  # Change 12, 1, 2 -> 1 #winter
    np.isin(most_common_month_dp, [3, 4, 5]),   # Change 3, 4, 5 -> 2 #spring
    np.isin(most_common_month_dp, [6, 7, 8]),   # Change 6, 7, 8 -> 3 #summer
    np.isin(most_common_month_dp, [9, 10, 11])  # Change 9, 10, 11 -> 4 #fall
]
conditions_pd = [
    np.isin(most_common_month_pd, [12, 1, 2]),  # Change 12, 1, 2 -> 1 #winter
    np.isin(most_common_month_pd, [3, 4, 5]),   # Change 3, 4, 5 -> 2 #spring
    np.isin(most_common_month_pd, [6, 7, 8]),   # Change 6, 7, 8 -> 3 #summer
    np.isin(most_common_month_pd, [9, 10, 11])  # Change 9, 10, 11 -> 4 #fall
]

most_common_season_dp = xr.DataArray(np.select(conditions_dp, choices, default=most_common_month_dp), coords=most_common_month_dp.coords)
most_common_season_pd = xr.DataArray(np.select(conditions_pd, choices, default=most_common_month_pd), coords=most_common_month_pd.coords)

#########################################################################################
# Load in file (droughts/pluvials)
#########################################################################################
dirname= '/data2/bpuxley/droughts_and_pluvials/Events/most_common_month_droughts_pluvials.nc'
most_common_month = xr.open_dataset(dirname)

most_common_month_droughts = most_common_month.droughts_common_month
most_common_month_pluvials = most_common_month.pluvials_common_month

# Define conditions and corresponding values
choices = [1, 2, 3, 4]
conditions_droughts = [
    np.isin(most_common_month_droughts, [12, 1, 2]),  # Change 12, 1, 2 -> 1 #winter
    np.isin(most_common_month_droughts, [3, 4, 5]),   # Change 3, 4, 5 -> 2 #spring
    np.isin(most_common_month_droughts, [6, 7, 8]),   # Change 6, 7, 8 -> 3 #summer
    np.isin(most_common_month_droughts, [9, 10, 11])  # Change 9, 10, 11 -> 4 #fall
]
conditions_pluvials = [
    np.isin(most_common_month_pluvials, [12, 1, 2]),  # Change 12, 1, 2 -> 1 #winter
    np.isin(most_common_month_pluvials, [3, 4, 5]),   # Change 3, 4, 5 -> 2 #spring
    np.isin(most_common_month_pluvials, [6, 7, 8]),   # Change 6, 7, 8 -> 3 #summer
    np.isin(most_common_month_pluvials, [9, 10, 11])  # Change 9, 10, 11 -> 4 #fall
]

most_common_season_droughts = xr.DataArray(np.select(conditions_droughts, choices, default=most_common_month_droughts), coords=most_common_month_droughts.coords)
most_common_season_pluvials = xr.DataArray(np.select(conditions_pluvials, choices, default=most_common_month_pluvials), coords=most_common_month_pluvials.coords)

#########################################################################################
# Read in Cluster Data
#########################################################################################
cluster_no = 7
cluster_polys = pd.read_csv('/data2/bpuxley/Events/cluster_polygons_final.csv')

cluster_polys['avg_poly'] = cluster_polys['avg_poly'].apply(shapely.wkt.loads)

#########################################################################################
# Create CONUS mask
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
event_freq_droughts_masked = np.where(mask, event_freq_droughts, np.nan)  # Set values outside the USA to NaN
event_freq_pluvials_masked = np.where(mask, event_freq_pluvials, np.nan)  # Set values outside the USA to NaN

most_common_month_dp_masked = np.where(mask, most_common_month_dp, np.nan)
most_common_month_pd_masked = np.where(mask, most_common_month_pd, np.nan)
most_common_month_droughts_masked = np.where(mask, most_common_month_droughts, np.nan)
most_common_month_pluvials_masked = np.where(mask, most_common_month_pluvials, np.nan)

most_common_season_dp_masked = np.where(mask, most_common_season_dp, np.nan)
most_common_season_pd_masked = np.where(mask, most_common_season_pd, np.nan)
most_common_season_droughts_masked = np.where(mask, most_common_season_droughts, np.nan)
most_common_season_pluvials_masked = np.where(mask, most_common_season_pluvials, np.nan)

#########################################################################################
# Plot - event frequency (all) (Figure 5)
#########################################################################################
fig = plt.figure(figsize = (11,7), dpi = 300)
gs = gridspec.GridSpec(6, 4, height_ratios=[0.45, 0.05, 0.12, 0.45, 0.05, 0.12],  width_ratios=[0.04, 0.46, 0.46, 0.04], hspace=0)

#Drought-to-Pluvial
ax1 = fig.add_subplot(gs[0,0:2], projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs1 = plt.contourf(lons, lats, event_freq_dp_masked, transform=ccrs.PlateCarree(), levels=np.arange(0, 91, 1), cmap = 'YlGnBu') 

plt.title("a) Drought-to-Pluvial Events", loc = 'left', fontsize = 10)

#Pluvial-to-Drought
ax2 = fig.add_subplot(gs[0,2:4], projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linewidth=1)
ax2.add_feature(cfeature.STATES, edgecolor='black')

ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs2 = plt.contourf(lons, lats, event_freq_pd_masked, transform=ccrs.PlateCarree(), levels=np.arange(0, 91, 1), cmap = 'YlGnBu') 

plt.title("b) Pluvial-to-Drought Events", loc = 'left', fontsize = 10)

# Colorbar under top row
cax1 = fig.add_subplot(gs[1, 1:3])
cb1 = fig.colorbar(cs1, cax=cax1, orientation='horizontal')
cb1.set_label('Count')

#Droughts
ax3 = fig.add_subplot(gs[3,0:2], projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax3.add_feature(cfeature.COASTLINE)
ax3.add_feature(cfeature.BORDERS, linewidth=1)
ax3.add_feature(cfeature.STATES, edgecolor='black')

ax3.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs3 = plt.contourf(lons, lats, event_freq_droughts_masked, transform=ccrs.PlateCarree(), levels=np.arange(0, 141, 1), cmap = 'YlGnBu') 

plt.title("c) Drought Events", loc = 'left', fontsize = 10)

#Pluvials
ax4 = fig.add_subplot(gs[3,2:4], projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax4.add_feature(cfeature.COASTLINE)
ax4.add_feature(cfeature.BORDERS, linewidth=1)
ax4.add_feature(cfeature.STATES, edgecolor='black')

ax4.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs4 = plt.contourf(lons, lats, event_freq_pluvials_masked, transform=ccrs.PlateCarree(), levels=np.arange(0, 141, 1), cmap = 'YlGnBu') 

plt.title("d) Pluvial Events", loc = 'left', fontsize = 10)

# Colorbar under bottom row
cax2 = fig.add_subplot(gs[4, 1:3])
cb2 = fig.colorbar(cs3, cax=cax2, orientation='horizontal', ticks=np.arange(0, 150, 10))
cb2.set_label('Count')

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_frequency_CONUS_all.png', bbox_inches = 'tight', pad_inches = 0.1)    

#########################################################################################
#Plot - event climatology (all) (Figure 6)
#########################################################################################
fig = plt.figure(figsize = (11,7), dpi = 300)
gs = gridspec.GridSpec(5, 4, height_ratios=[0.45, 0.01, 0.45, 0.03, 0.12],  width_ratios=[0.04, 0.46, 0.46, 0.04], hspace=0)

cmap = ListedColormap(["cornflowerblue","mediumseagreen","lightpink","sandybrown"]) #season

#Drought-to-Pluvial
ax1 = fig.add_subplot(gs[0,0:2], projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs1 = plt.contourf(lons, lats, most_common_season_dp_masked, transform=ccrs.PlateCarree(), levels=np.arange(0, 5, 1), cmap=cmap) 

plt.title("a) Drought-to-Pluvial Events", loc = 'left', fontsize = 10)

#Pluvial-to-Drought
ax2 = fig.add_subplot(gs[0,2:4], projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linewidth=1)
ax2.add_feature(cfeature.STATES, edgecolor='black')

ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs2 = plt.contourf(lons, lats, most_common_season_pd_masked, transform=ccrs.PlateCarree(), levels=np.arange(0, 5, 1), cmap=cmap) 

plt.title("b) Pluvial-to-Drought Events", loc = 'left', fontsize = 10)

#Droughts
ax3 = fig.add_subplot(gs[2,0:2], projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax3.add_feature(cfeature.COASTLINE)
ax3.add_feature(cfeature.BORDERS, linewidth=1)
ax3.add_feature(cfeature.STATES, edgecolor='black')

ax3.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs3 = plt.contourf(lons, lats, most_common_season_droughts_masked, transform=ccrs.PlateCarree(), levels=np.arange(0, 5, 1), cmap=cmap) 

plt.title("c) Drought Events", loc = 'left', fontsize = 10)

#Pluvials
ax4 = fig.add_subplot(gs[2,2:4], projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax4.add_feature(cfeature.COASTLINE)
ax4.add_feature(cfeature.BORDERS, linewidth=1)
ax4.add_feature(cfeature.STATES, edgecolor='black')

ax4.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs4 = plt.contourf(lons, lats, most_common_season_pluvials_masked, transform=ccrs.PlateCarree(), levels=np.arange(0, 5, 1), cmap=cmap) 

plt.title("d) Pluvial Events", loc = 'left', fontsize = 10)

# Colorbar under bottom row
cax2 = fig.add_subplot(gs[3, 1:3])
cb = fig.colorbar(cs3, cax=cax2, orientation='horizontal', ticks=np.arange(0, 150, 10))
cb.set_label('Season')
cb.set_ticks(np.arange(0.5,4.5,1))
cb.set_ticklabels(['DJF (winter)','MAM (spring)','JJA (summer)',' SON (fall)'])

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_climatology_CONUS_all.png', bbox_inches = 'tight', pad_inches = 0.1)    

""""
Extra Code
#########################################################################################
#Plot - event frequency
#########################################################################################
colors = ['red', 'green', 'purple', 'saddlebrown', 'blue', 'black', 'orange', 'gold', 'pink', 'deeppink', 
			'deepskyblue', 'springgreen', 'olive', 'tan', 'grey', 'darkred', 'cyan', 'mediumpurple','tomato','chocolate',
			'yellow','lawngreen','lavender','plum','fuchsia','palevioletred','rosybrown','darkcyan','aquamarine','navy']


fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

#Drought-to-Pluvial
ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

for i in range(0, cluster_no):
	print(i)
	color = colors[i]
	print(color)
	polygons = cluster_polys['avg_poly'][i]
	x, y = polygons.exterior.xy
	ax1.plot(x, y, transform=ccrs.PlateCarree(), color=color)
	
	# Get centroid and add label
	centroid = polygons.centroid
	cx, cy = centroid.x, centroid.y
	
	#ax1.text(cx, cy, str(i+1), transform=ccrs.PlateCarree(),
			#ha='center', va='center', fontsize=10,
			#fontweight='bold', color=color,
			#bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1', alpha=1))		

cs = plt.contourf(lons, lats, event_freq_dp_masked, transform=ccrs.PlateCarree(), levels=np.arange(0, 91, 1), cmap = 'YlGnBu') 

plt.title("Total Number of Times a Grid Point \nis in an Event Polygon", loc = "left")
plt.title("Drought-to-Pluvial", loc = 'right')

fig.colorbar(cs, ax=ax1, orientation='horizontal', pad=0.05)

#Pluvial-to-Drought
ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linewidth=1)
ax2.add_feature(cfeature.STATES, edgecolor='black')

ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

for i in range(0, cluster_no):
	print(i)
	color = colors[i]
	print(color)
	polygons = cluster_polys['avg_poly'][i]
	x, y = polygons.exterior.xy
	ax2.plot(x, y, transform=ccrs.PlateCarree(), color=color)
	
	# Get centroid and add label
	centroid = polygons.centroid
	cx, cy = centroid.x, centroid.y
	
	#ax2.text(cx, cy, str(i+1), transform=ccrs.PlateCarree(),
			#ha='center', va='center', fontsize=10,
			#fontweight='bold', color=color,
			#bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1', alpha=1))		

cs = plt.contourf(lons, lats, event_freq_pd_masked, transform=ccrs.PlateCarree(), levels=np.arange(0, 91, 1), cmap = 'YlGnBu') 

plt.title("Total Number of Times a Grid Point \nis in an Event Polygon", loc = "left")
plt.title("Pluvial-to-Drought", loc = 'right')

fig.colorbar(cs, ax=ax2, orientation='horizontal', pad=0.05)
	
plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_frequency_CONUS_day0only_withpolys.png', bbox_inches = 'tight', pad_inches = 0.1)    

#########################################################################################
#Plot - event monthly climatology
#########################################################################################
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

cmap = ListedColormap(["cornflowerblue","mediumseagreen","lightpink","sandybrown"]) #season
#cmap = ListedColormap(["cornflowerblue","lightsteelblue","green","mediumseagreen","mediumaquamarine","mediumvioletred","palevioletred","lightpink","saddlebrown","chocolate", "sandybrown","royalblue"]) #monthly

#Drought-to-Pluvial
ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

for i in range(0, cluster_no):
	print(i)
	color = colors[i]
	print(color)
	polygons = cluster_polys['avg_poly'][i]
	x, y = polygons.exterior.xy
	ax1.plot(x, y, transform=ccrs.PlateCarree(), color=color)
	
	# Get centroid and add label
	centroid = polygons.centroid
	cx, cy = centroid.x, centroid.y
	
	#ax1.text(cx, cy, str(i+1), transform=ccrs.PlateCarree(),
			#ha='center', va='center', fontsize=10,
			#fontweight='bold', color=color,
			#bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1', alpha=1))		

cs = plt.contourf(lons, lats, most_common_season_dp_masked, transform=ccrs.PlateCarree(), levels=np.arange(0, 5, 1), cmap=cmap) 

plt.title("Most Common Season a Grid Point \nis in an Event Polygon", loc = "left")
plt.title("Drought-to-Pluvial", loc = 'right')

# Add colorbar with custom ticks
cbar = plt.colorbar(cs, ax=ax1, orientation='horizontal', pad=0.05)
#cbar.set_ticks(np.arange(0.5,12.5,1))
#cbar.set_ticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

cbar.set_ticks(np.arange(0.5,4.5,1))
cbar.set_ticklabels(['DJF (winter)','MAM (spring)','JJA (summer)',' SON (fall)'])

#Pluvial-to-Drought
ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linewidth=1)
ax2.add_feature(cfeature.STATES, edgecolor='black')

ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

for i in range(0, cluster_no):
	print(i)
	color = colors[i]
	print(color)
	polygons = cluster_polys['avg_poly'][i]
	x, y = polygons.exterior.xy
	ax2.plot(x, y, transform=ccrs.PlateCarree(), color=color)
	
	# Get centroid and add label
	centroid = polygons.centroid
	cx, cy = centroid.x, centroid.y
	
	#ax2.text(cx, cy, str(i+1), transform=ccrs.PlateCarree(),
			#ha='center', va='center', fontsize=10,
			#fontweight='bold', color=color,
			#bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1', alpha=1))		


cs = plt.contourf(lons, lats, most_common_season_pd_masked, transform=ccrs.PlateCarree(), levels=np.arange(0, 5, 1), cmap=cmap) 

plt.title("Most Common Season a Grid Point \nis in an Event Polygon", loc = "left")
plt.title("Pluvial-to-Drought", loc = 'right')

# Add colorbar with custom ticks
cbar = plt.colorbar(cs, ax=ax2, orientation='horizontal', pad=0.05)
#cbar.set_ticks(np.arange(0.5,12.5,1))
#cbar.set_ticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

cbar.set_ticks(np.arange(0.5,4.5,1))
cbar.set_ticklabels(['DJF (winter)','MAM (spring)','JJA (summer)',' SON (fall)'])

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_climatology_CONUS_day0only_season_withpolys.png', bbox_inches = 'tight', pad_inches = 0.1)    







