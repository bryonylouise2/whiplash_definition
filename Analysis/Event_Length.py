#########################################################################################
## A script to calculate how long an event was in drought/pluvial conditions prior to and 
## after the event
## Bryony Louise
## Last Edited: Thursday February 27th 2025 
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

def calc_spi_avg(lons, lats, poly, spi):
	# mask to only event region
	mask = functions._mask_outside_region(lons, lats, poly)
	
	spi = np.ma.masked_array(spi, ~mask)
	
	# mask any nans that slipped through
	spi_new = np.ma.masked_invalid(spi)  
	
	# area-avg spi
	weights = np.cos(np.radians(lats[:, 0]))
	spi_tmp = np.ma.mean(spi_new, axis=-1)  # avg across lons first
	spi_area_avg = np.ma.average(spi_tmp, weights=weights)
	
	return spi_area_avg

#########################################################################################
#Import Events - load previously made event files
#########################################################################################
events_DP = pd.read_csv('/data2/bpuxley/Events/independent_events_DP.csv')
events_PD = pd.read_csv('/data2/bpuxley/Events/independent_events_PD.csv')

df_DP = events_DP.copy()
df_PD = events_PD.copy()

#########################################################################################
#Read in SPI Data
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

#Lons, lats
lons, lats = np.meshgrid(spi.lon.values, spi.lat.values) #create a meshgrid of lat,lon values

#########################################################################################
#Create a list of "start" and "end" dates of events (based on whiplash date)
#########################################################################################
#Start Dates
start_dates_DP = functions.start_dates(df_DP)
start_dates_PD = functions.start_dates(df_PD)

#End Dates and Polygons
end_dates_DP, polygons_DP = functions.end_dates_and_polys(df_DP)
end_dates_PD, polygons_PD = functions.end_dates_and_polys(df_PD)

#centroids = [poly.centroid for poly in polygons] #get centroid of polygons

#########################################################################################
#Drought-to-Pluvial
#########################################################################################
#########################################################################################
#How long in drought conditions prior?
#########################################################################################
spi_before = []
spi_before_num = []
for i,(dates,polys) in tqdm(enumerate(zip(start_dates_DP, polygons_DP))):
	event_num = i+1
	spi_prior = []
	event_length = len(np.where((df_DP.Event_No == event_num))[0])
	for d in range(0, 1000):
		analysis_date = (pd.to_datetime(dates) + timedelta(days=-d)).strftime('%Y-%m-%d')
		avg_spi = calc_spi_avg(lons, lats, polys, spi.spi_30day.sel(time=analysis_date))
	
		if avg_spi <= 0:
			spi_prior.append(avg_spi)
		else:
			break
	
	spi_before_num += ([len(spi_prior)]*event_length)	
	spi_before += ([spi_prior]*event_length)
	
df_DP['SPI_Values_Before'] = np.nan
df_DP['SPI_Values_Before'] = df_DP['SPI_Values_Before'].astype('object')
df_DP['SPI_Values_Before'] = spi_before

df_DP['Time_in_Drought_Before'] = pd.DataFrame(spi_before_num)

#########################################################################################
#How long in pluvial condiitons after?
#########################################################################################
spi_after = []
spi_after_num = []
for i,(dates,polys) in tqdm(enumerate(zip(end_dates_DP, polygons_DP))):
	event_num = i+1
	spi_posterior = []
	event_length = len(np.where((df_DP.Event_No == event_num))[0])
	for d in range(0, 1000):
		analysis_date = (pd.to_datetime(dates) + timedelta(days=d+30)).strftime('%Y-%m-%d')
		if analysis_date[0:4] == '2021':
			break
		else:
			avg_spi = calc_spi_avg(lons, lats, polys, spi.spi_30day.sel(time=analysis_date))
	
			if avg_spi >= 0:
				spi_posterior.append(avg_spi)
			else:
				break
	
	spi_after_num += ([len(spi_posterior)]*event_length)	
	spi_after += ([spi_posterior]*event_length)
	
df_DP['SPI_Values_After'] = np.nan
df_DP['SPI_Values_After'] = df_DP['SPI_Values_After'].astype('object')
df_DP['SPI_Values_After'] = spi_after

df_DP['Time_in_Pluvial_After'] = pd.DataFrame(spi_after_num)

#########################################################################################
#Save out file
#########################################################################################
df_DP.to_csv(f'/data2/bpuxley/Events/independent_events_DP.csv', index=False)
    
#########################################################################################
#Pluvial-to-Drought
#########################################################################################
#########################################################################################
#How long in pluvial conditions prior?
#########################################################################################
spi_before = []
spi_before_num = []
for i,(dates,polys) in tqdm(enumerate(zip(start_dates_PD, polygons_PD))):
	event_num = i+1
	spi_prior = []
	event_length = len(np.where((df_PD.Event_No == event_num))[0])
	for d in range(0, 1000):
		analysis_date = (pd.to_datetime(dates) + timedelta(days=-d)).strftime('%Y-%m-%d')
		avg_spi = calc_spi_avg(lons, lats, polys, spi.spi_30day.sel(time=analysis_date))
	
		if avg_spi >= 0:
			spi_prior.append(avg_spi)
		else:
			break
	
	spi_before_num += ([len(spi_prior)]*event_length)	
	spi_before += ([spi_prior]*event_length)
	
df_PD['SPI_Values_Before'] = np.nan
df_PD['SPI_Values_Before'] = df_PD['SPI_Values_Before'].astype('object')
df_PD['SPI_Values_Before'] = spi_before

df_PD['Time_in_Pluvial_Before'] = pd.DataFrame(spi_before_num)

#########################################################################################
#How long in drought conditions after?
#########################################################################################
spi_after = []
spi_after_num = []
for i,(dates,polys) in tqdm(enumerate(zip(end_dates_PD, polygons_PD))):
	event_num = i+1
	spi_posterior = []
	event_length = len(np.where((df_PD.Event_No == event_num))[0])
	for d in range(0, 1000):
		analysis_date = (pd.to_datetime(dates) + timedelta(days=d+30)).strftime('%Y-%m-%d')
		if analysis_date[0:4] == '2021':
			break
		else:
			avg_spi = calc_spi_avg(lons, lats, polys, spi.spi_30day.sel(time=analysis_date))
	
			if avg_spi <= 0:
				spi_posterior.append(avg_spi)
			else:
				break
	
	spi_after_num += ([len(spi_posterior)]*event_length)	
	spi_after += ([spi_posterior]*event_length)
	
df_PD['SPI_Values_After'] = np.nan
df_PD['SPI_Values_After'] = df_PD['SPI_Values_After'].astype('object')
df_PD['SPI_Values_After'] = spi_after

df_PD['Time_in_Drought_After'] = pd.DataFrame(spi_after_num)

#########################################################################################
#Add a seasonal column
#########################################################################################
df_DP['month'] = pd.to_datetime(df_DP.Whiplash_Date).dt.month
df_PD['month']  = pd.to_datetime(df_PD.Whiplash_Date).dt.month

def categorize_season(value):
	if value == 12 or value == 1 or value == 2:
		return 1
	if value == 3 or value == 4 or value == 5:
		return 2
	if value == 6 or value == 7 or value == 8:
		return 3
	else:
		return 4
		
df_DP['season'] = df_DP['month'].apply(categorize_season)
df_PD['season'] = df_PD['month'].apply(categorize_season)
		

#########################################################################################
#Save out file
#########################################################################################
df_PD.to_csv(f'/data2/bpuxley/Events/independent_events_PD.csv', index=False)

#########################################################################################
#Plot as a histogram
#########################################################################################
fig = plt.figure(figsize = (10,5), dpi = 300, tight_layout =True)

bins = np.arange(0, 305, 10)

#Drought-to-Pluvial
ax1 = fig.add_subplot(221)

plt.hist(df_DP.Time_in_Drought_Before, bins, density = False, histtype ='bar',color='mediumpurple', edgecolor ='k', linewidth=0.5)
plt.xticks(np.arange(0, 301, 10))
ax1.set_xticklabels(['0','','20','','40','','60','','80','','100','','120','','140','','160','','180','','200','','220','','240','','260','','280','','300'], fontsize=7)
plt.yticks(np.arange(0, 1500, 250))
ax1.set_yticklabels(['0','250','500', '750','1,000', '1,250'], fontsize=7)

ax1.set_xlabel('No. of 30-day SPI rolling periods below 0 prior to whiplash event', fontsize = 10)
ax1.set_ylabel('Number of Events', fontsize = 10)
plt.title("a) Time in Drought prior to Drought-to-Pluvial Whiplash", fontsize=12)

ax2 = fig.add_subplot(222)

plt.hist(df_DP.Time_in_Pluvial_After, bins, density = False, histtype ='bar',color='mediumpurple', edgecolor ='k', linewidth=0.5)
plt.xticks(np.arange(0, 301, 10))
ax2.set_xticklabels(['0','','20','','40','','60','','80','','100','','120','','140','','160','','180','','200','','220','','240','','260','','280','','300'], fontsize=7)
plt.yticks(np.arange(0, 1500, 250))
ax2.set_yticklabels(['0','250','500', '750','1,000', '1,250'], fontsize=7)

ax2.set_xlabel('No. of 30-day SPI rolling periods above 0 after whiplash event', fontsize = 10)
ax2.set_ylabel('Number of Events', fontsize = 10)
plt.title("b) Time in Pluvial after Drought-to-Pluvial Whiplash", fontsize=12)

#Pluvial-to-Drought
ax3 = fig.add_subplot(223)

plt.hist(df_PD.Time_in_Pluvial_Before, bins, density = False, histtype ='bar',color='lightcoral', edgecolor ='k', linewidth=0.5)
plt.xticks(np.arange(0, 301, 10))
ax3.set_xticklabels(['0','','20','','40','','60','','80','','100','','120','','140','','160','','180','','200','','220','','240','','260','','280','','300'], fontsize=7)
plt.yticks(np.arange(0, 1500, 250))
ax3.set_yticklabels(['0','250','500', '750','1,000', '1,250'], fontsize=7)

ax3.set_xlabel('No. of 30-day SPI rolling periods above 0 prior to whiplash event', fontsize = 10)
ax3.set_ylabel('Number of Events', fontsize = 10)
plt.title("c) Time in Pluvial prior to Pluvial-to-Drought Whiplash", fontsize=12)

ax4 = fig.add_subplot(224)

plt.hist(df_PD.Time_in_Drought_After, bins, density = False, histtype ='bar',color='lightcoral', edgecolor ='k', linewidth=0.5)
plt.xticks(np.arange(0, 301, 10))
ax4.set_xticklabels(['0','','20','','40','','60','','80','','100','','120','','140','','160','','180','','200','','220','','240','','260','','280','','300'], fontsize=7)
plt.yticks(np.arange(0, 1500, 250))
ax4.set_yticklabels(['0','250','500', '750','1,000', '1,250'], fontsize=7)

ax4.set_xlabel('No. of 30-day SPI rolling periods below 0 after whiplash event', fontsize = 10)
ax4.set_ylabel('Number of Events', fontsize = 10)
plt.title("d) Time in Drought after Pluvial-to-Drought Whiplash", fontsize=12)

plt.tight_layout()
#plt.show(block=False)

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_length.png', bbox_inches = 'tight', pad_inches = 0.1)    

#########################################################################################
#Scatter Plot with histogram
#########################################################################################
fig = plt.figure(figsize = (10,5), dpi = 300, tight_layout =True)

#day_0_only
df_DP_new = df_DP[df_DP.Day_No ==0].reset_index(drop=True) 
df_PD_new = df_PD[df_PD.Day_No ==0].reset_index(drop=True) 

df_DP_new = df_DP_new[df_DP_new.Time_in_Pluvial_After <=100].reset_index(drop=True) 
df_DP_new = df_DP_new[df_DP_new.Time_in_Drought_Before <=100].reset_index(drop=True) 

df_PD_new = df_PD_new[df_PD_new.Time_in_Pluvial_Before <=100].reset_index(drop=True) 
df_PD_new = df_PD_new[df_PD_new.Time_in_Drought_After <=100].reset_index(drop=True) 

sizes_DP = (df_DP_new.Area/np.nanmax(df_DP_new.Area))*10
sizes_PD = (df_PD_new.Area/np.nanmax(df_PD_new.Area))*10

colors_DP = df_DP_new.season
colors_PD = df_PD_new.season

#Drought-to-Pluvial
ax1 = fig.add_subplot(121)
plt.scatter(df_DP_new.Time_in_Drought_Before, df_DP_new.Time_in_Pluvial_After, s=sizes_DP, c=colors_DP,cmap='gist_rainbow')
plt.xticks(np.arange(0, 110, 20))
ax1.set_xticklabels(['0','20','40','60','80','100'], fontsize=7)
plt.yticks(np.arange(0, 110, 20))
ax1.set_yticklabels(['0','20','40','60','80','100'], fontsize=7)

ax1.set_xlabel('Time in Drought Before', fontsize = 10)
ax1.set_ylabel('Time in Pluvial After', fontsize = 10)	

cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=7) 
cbar.ax.set_title('Month of the Year',fontsize=7)	
plt.title("a) Drought-to-Pluvial", fontsize=12)

#Pluvial-to-Drought
ax2 = fig.add_subplot(122)
plt.scatter(df_PD_new.Time_in_Pluvial_Before, df_PD_new.Time_in_Drought_After,s=sizes_PD, c=colors_PD,cmap='gist_rainbow')
plt.xticks(np.arange(0, 110, 20))
ax2.set_xticklabels(['0','20','40','60','80','100'], fontsize=7)
plt.yticks(np.arange(0, 110, 20))
ax2.set_yticklabels(['0','20','40','60','80','100'], fontsize=7)

ax2.set_xlabel('Time in Pluvial Before', fontsize = 10)
ax2.set_ylabel('Time in Drought After', fontsize = 10)

cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=7) 
cbar.ax.set_title('Month of the Year',fontsize=7)	
plt.title("a) Pluvial-to-Drought", fontsize=12)

plt.tight_layout()
#plt.show(block=False)

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_length_scatter_season_small.png', bbox_inches = 'tight', pad_inches = 0.1)    





