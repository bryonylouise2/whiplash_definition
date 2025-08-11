#########################################################################################
## Examine and Plot Specific Events
## Bryony Louise Puxley
## Last Edited: Monday, August 11th, 2025
## Input: Independent event files of Drought-to-Pluvial and Pluvial-to-Drought events.
## Previously made SPI, whiplash occurrence, and normalized density files, 
## Output: Two PNG files. Figure 1 from the journal article: 30-day SPI during (a) the 
## drought period, and (b) the following pluvial period. (c) Points flagged as having a 
## whiplash event as per our definition (see the text for details). (d) Full KDE normalized 
## density field using the Epanechnikov kernel and 0.02 bandwidth. In red on all subplots, 
## the event polygon is drawn using the 0.4878 contour. Figure S1 from the journal article: 
## the same for pluvial-to-drought.
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

#########################################################################################
# Choose Precipitation Whiplash Type and run the corresponding cell
#########################################################################################
#########################################################################################
# Drought-to-Pluvial
#########################################################################################
event_num =  486 # choose the event number for examination

#Event Databases from 10.Independence Algorithm
events_DP = pd.read_csv('/data2/bpuxley/Events/independent_events_DP.csv')
df = events_DP.copy()

#########################################################################################
# Pluvial-to-Drought
#########################################################################################
event_num =  715 # choose the event number for examination

#Event Databases from 10.Independence Algorithm
events_PD = pd.read_csv('/data2/bpuxley/Events/independent_events_PD.csv')
df = events_PD.copy()

#########################################################################################
# Import all other necessary data - load previously made SPI, whiplash identification, and 
# normalized density files
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

#Normalized Density from 6.Spatial_Consistency
dirname= '/data2/bpuxley/Density'

pathfiles = []

for i in years:
	filename = 'density_%s.nc'%(i)
	pathfile = os.path.join(dirname, filename)
	pathfiles.append(pathfile)

density = xr.open_mfdataset(pathfiles, combine='by_coords')

print('Read in Data')

#Lons, lats
lons, lats = np.meshgrid(whiplashes.lon.values, whiplashes.lat.values) #create a meshgrid of lat,lon values

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
whiplash_dates = event[third_column]

#length of event
event_len = len(event)

##########################################################################################
# Plot the event - choose the correct whiplash type
##########################################################################################
##########################################################################################
## DROUGHT TO PLUVIAL
##########################################################################################
day_num = 0 #day within the event for examination

whiplash_date_0 = (event.iloc[np.where((event.Day_No == day_num))].Whiplash_Date).values[0]
whiplash_date_30 = (pd.to_datetime(whiplash_date_0) + timedelta(days=30)).strftime('%Y-%m-%d')
drought_date = event.iloc[np.where((event.Day_No == day_num))].Drought_Date.values[0]
pluvial_date = event.iloc[np.where((event.Day_No == day_num))].Pluvial_Date.values[0]
poly = polygons[day_num]

#Mask Data to CONUS
spi_drought_masked = np.where(mask,  spi.spi_30day.sel(time=whiplash_date_0).values, np.nan) # Set values outside the USA to NaN
spi_pluvial_masked = np.where(mask, spi.spi_30day.sel(time=whiplash_date_30).values, np.nan) # Set values outside the USA to NaN
whiplash_masked = np.where(mask, whiplashes.DP_whiplashes.sel(time=whiplash_date_0), np.nan) # Set values outside the USA to NaN
KDE_masked = np.where(mask, density.DP_density.sel(time=whiplash_date_0), np.nan) # Set values outside the USA to NaN

#Plot
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

# First subplot of SPI drought values (CONUS)
ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

poly_x, poly_y = poly.exterior.xy
plt.plot(poly_x, poly_y, color='red', linewidth=2.5, transform=ccrs.PlateCarree())
		
cs1 = plt.contourf(lons, lats, spi_drought_masked, transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-3, 3, 0.1), cmap = 'BrBG') 

fig.colorbar(cs1, ax=ax1, orientation='horizontal', pad=0.05)

plt.title("a) SPI CONUS \nDate: %s"%(drought_date), loc = "left")
plt.title("Event No: %s\nDay No: %s"%(event_num, day_num), loc = 'right')

# Second subplot of SPI pluvial values (CONUS)
ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linewidth=1)
ax2.add_feature(cfeature.STATES, edgecolor='black')

ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

poly_x, poly_y = poly.exterior.xy
plt.plot(poly_x, poly_y, color='red', linewidth=2.5, transform=ccrs.PlateCarree())
		
cs2 = plt.contourf(lons, lats, spi_pluvial_masked, transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-3, 3, 0.1), cmap = 'BrBG') 

fig.colorbar(cs2, ax=ax2, orientation='horizontal', pad=0.05)

plt.title("b) SPI CONUS \nDate: %s"%(pluvial_date), loc = "left")
plt.title("Event No: %s\nDay No: %s"%(event_num, day_num), loc = 'right')

#Third subplot of whiplash locations (CONUS)
ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax3.add_feature(cfeature.COASTLINE)
ax3.add_feature(cfeature.BORDERS, linewidth=1)
ax3.add_feature(cfeature.STATES, edgecolor='black')

ax3.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

poly_x, poly_y = poly.exterior.xy
plt.plot(poly_x, poly_y, color='red', linewidth=2.5, transform=ccrs.PlateCarree())
		
# Create a custom color map: purple for True, white for False
cmap = ListedColormap(["white", "purple"])

cs3 = ax3.pcolormesh(lons, lats, whiplash_masked, cmap=cmap, transform=ccrs.PlateCarree())

#fig.colorbar(cs3, ax=ax3, orientation='horizontal', pad=0.05)

plt.title('c) Whiplash Occurrences', loc ="left")
plt.title("Event No: %s\nDay No: %s"%(event_num, day_num), loc = 'right')

# Fourth subplot with the normalized density
ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax4.add_feature(cfeature.COASTLINE)
ax4.add_feature(cfeature.BORDERS, linewidth=1)
ax4.add_feature(cfeature.STATES, edgecolor='black')

ax4.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

poly_x, poly_y = poly.exterior.xy
plt.plot(poly_x, poly_y, color='red', linewidth=2.5, transform=ccrs.PlateCarree())

cs4 = plt.contourf(lons, lats, KDE_masked, transform=ccrs.PlateCarree(), levels=np.arange(0, 1.1, 0.1), cmap = 'Purples') 

fig.colorbar(cs4, ax=ax4, orientation='horizontal', pad=0.05)

plt.title('d) Normalized Density', loc="left")
plt.title("Event No: %s\nDay No: %s"%(event_num, day_num), loc = 'right')

#plt.show(block=False)

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/Events_Drought_to_Pluvial/event_%s_%s.png'%(event_num,day_num), bbox_inches = 'tight', pad_inches = 0.1)    
 
##########################################################################################
## PLUVIAL TO DROUGHT
##########################################################################################
day_num = 0 #day within the event for examination

whiplash_date_0 = (event.iloc[np.where((event.Day_No == day_num))].Whiplash_Date).values[0]
whiplash_date_30 = (pd.to_datetime(whiplash_date_0) + timedelta(days=30)).strftime('%Y-%m-%d')
drought_date = event.iloc[np.where((event.Day_No == day_num))].Drought_Date.values[0]
pluvial_date = event.iloc[np.where((event.Day_No == day_num))].Pluvial_Date.values[0]
poly = polygons[day_num]

#Mask Data to CONUS
spi_pluvial_masked = np.where(mask,  spi.spi_30day.sel(time=whiplash_date_0).values, np.nan) # Set values outside the USA to NaN
spi_drought_masked = np.where(mask, spi.spi_30day.sel(time=whiplash_date_30).values, np.nan) # Set values outside the USA to NaN
whiplash_masked = np.where(mask, whiplashes.PD_whiplashes.sel(time=whiplash_date_0), np.nan) # Set values outside the USA to NaN
KDE_masked = np.where(mask, density.PD_density.sel(time=whiplash_date_0), np.nan) # Set values outside the USA to NaN

#Plot
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

# First subplot of SPI pluvial values (CONUS)
ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

poly_x, poly_y = poly.exterior.xy
plt.plot(poly_x, poly_y, color='red', linewidth=2.5, transform=ccrs.PlateCarree())
		
cs1 = plt.contourf(lons, lats, spi_pluvial_masked, transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-3, 3, 0.1), cmap = 'BrBG') 

fig.colorbar(cs1, ax=ax1, orientation='horizontal', pad=0.05)

plt.title("a) SPI CONUS \nDate: %s"%(pluvial_date), loc = "left")
plt.title("Event No: %s\nDay No: %s"%(event_num, day_num), loc = 'right')

# Second subplot of SPI pluvial values (CONUS)
ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linewidth=1)
ax2.add_feature(cfeature.STATES, edgecolor='black')

ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

poly_x, poly_y = poly.exterior.xy
plt.plot(poly_x, poly_y, color='red', linewidth=2.5, transform=ccrs.PlateCarree())
		
cs2 = plt.contourf(lons, lats, spi_drought_masked, transform=ccrs.PlateCarree(), extend = "both", levels=np.arange(-3, 3, 0.1), cmap = 'BrBG') 

fig.colorbar(cs2, ax=ax2, orientation='horizontal', pad=0.05)

plt.title("b) SPI CONUS \nDate: %s"%(drought_date), loc = "left")
plt.title("Event No: %s\nDay No: %s"%(event_num, day_num), loc = 'right')

#Third subplot of whiplash locations (CONUS)
ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax3.add_feature(cfeature.COASTLINE)
ax3.add_feature(cfeature.BORDERS, linewidth=1)
ax3.add_feature(cfeature.STATES, edgecolor='black')

ax3.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

poly_x, poly_y = poly.exterior.xy
plt.plot(poly_x, poly_y, color='red', linewidth=2.5, transform=ccrs.PlateCarree())
		
# Create a custom color map: purple for True, white for False
cmap = ListedColormap(["white", "purple"])

cs3 = ax3.pcolormesh(lons, lats, whiplash_masked, cmap=cmap, transform=ccrs.PlateCarree())

#fig.colorbar(cs3, ax=ax3, orientation='horizontal', pad=0.05)

plt.title('c) Whiplash Occurrences', loc ="left")
plt.title("Event No: %s\nDay No: %s"%(event_num, day_num), loc = 'right')

# Fourth subplot with the normalized density
ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax4.add_feature(cfeature.COASTLINE)
ax4.add_feature(cfeature.BORDERS, linewidth=1)
ax4.add_feature(cfeature.STATES, edgecolor='black')

ax4.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

poly_x, poly_y = poly.exterior.xy
plt.plot(poly_x, poly_y, color='red', linewidth=2.5, transform=ccrs.PlateCarree())

cs4 = plt.contourf(lons, lats, KDE_masked, transform=ccrs.PlateCarree(), levels=np.arange(0, 1.1, 0.1), cmap = 'Purples') 

fig.colorbar(cs4, ax=ax4, orientation='horizontal', pad=0.05)

plt.title('d) Normalized Density', loc="left")
plt.title("Event No: %s\nDay No: %s"%(event_num, day_num), loc = 'right')

#plt.show(block=False)

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/Events_Pluvial_to_Drought/event_%s_%s.png'%(event_num,day_num), bbox_inches = 'tight', pad_inches = 0.1)    


#########################################################################################
for i in range(0,len(event_index)):
	polygon = polygons_DP[event_index[i]] 
	poly_x, poly_y = polygon.exterior.xy
	
	if potential_events_DP.Area[event_index[i]] >= AREA:
		plt.plot(poly_x, poly_y, color='red', linewidth=2.5, transform=ccrs.PlateCarree())
	else:
		plt.plot(poly_x, poly_y, color='black', linewidth=2.5, transform=ccrs.PlateCarree())
