#########################################################################################
## A script to calculate how long an event was in drought/pluvial conditions prior to and 
## after the event
## Bryony Louise Puxley
## Last Edited: Monday, August 11th, 2025 
## Input: Independent event files of Drought-to-Pluvial, Pluvial-to-Drought, Drought only,
## and Pluvial only events. Decadal SPI fiLes. 
## Output: Updated independent event files of Drought-to-Pluvial, Pluvial-to-Drought, 
## Drought only, and Pluvial only events with columns for Time in *CONDITION* Before, and
## Time in *CONDITION* After for respective events.
#########################################################################################
# Import Required Modules
#########################################################################################
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

#########################################################################################
# Import Functions
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
# Import Events - load previously made event files
#########################################################################################
events_DP = pd.read_csv('/data2/bpuxley/Events/independent_events_DP.csv')
events_PD = pd.read_csv('/data2/bpuxley/Events/independent_events_PD.csv')
events_droughts = pd.read_csv('/data2/bpuxley/Events/independent_events_droughts.csv')
events_pluvials = pd.read_csv('/data2/bpuxley/Events/independent_events_pluvials.csv')

df_DP = events_DP.copy()
df_PD = events_PD.copy()
df_droughts = events_droughts.copy()
df_pluvials = events_pluvials.copy()

#########################################################################################
# Read in SPI Data
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
# Create a list of "start" and "end" dates of events (based on whiplash date)
#########################################################################################
#Start Dates
start_dates_DP = functions.start_dates(df_DP)
start_dates_PD = functions.start_dates(df_PD)
start_dates_droughts = functions.start_dates_droughts_pluvials(df_droughts)
start_dates_pluvials = functions.start_dates_droughts_pluvials(df_pluvials)

#End Dates and Polygons
end_dates_DP, polygons_DP = functions.end_dates_and_polys(df_DP)
end_dates_PD, polygons_PD = functions.end_dates_and_polys(df_PD)
end_dates_droughts, polygons_droughts = functions.end_dates_and_polys_droughts_pluvials(df_droughts)
end_dates_pluvials, polygons_pluvials = functions.end_dates_and_polys_droughts_pluvials(df_pluvials)

#########################################################################################
# Drought-to-Pluvial
#########################################################################################
#########################################################################################
# How long in drought conditions prior?
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
# How long in pluvial condiitons after?
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
# Save out file
#########################################################################################
df_DP.to_csv(f'/data2/bpuxley/Events/independent_events_DP.csv', index=False)
    
#########################################################################################
# Pluvial-to-Drought
#########################################################################################
#########################################################################################
# How long in pluvial conditions prior?
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
# How long in drought conditions after?
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
# Save out file
#########################################################################################
df_PD.to_csv(f'/data2/bpuxley/Events/independent_events_PD.csv', index=False)


#########################################################################################
# Drought Events
#########################################################################################
#########################################################################################
# How long in drought conditions prior?
#########################################################################################
spi_before = []
spi_before_num = []
for i,(dates,polys) in tqdm(enumerate(zip(start_dates_droughts, polygons_droughts))):
	event_num = i+1
	spi_prior = []
	event_length = len(np.where((df_droughts.Event_No == event_num))[0])
	for d in range(0, 1000):
		analysis_date = (pd.to_datetime(dates) + timedelta(days=-d)).strftime('%Y-%m-%d')
		avg_spi = calc_spi_avg(lons, lats, polys, spi.spi_30day.sel(time=analysis_date))
	
		if avg_spi <= 0:
			spi_prior.append(avg_spi)
		else:
			break
	
	spi_before_num += ([len(spi_prior)]*event_length)	
	spi_before += ([spi_prior]*event_length)
	
df_droughts['SPI_Values_Before'] = np.nan
df_droughts['SPI_Values_Before'] = df_droughts['SPI_Values_Before'].astype('object')
df_droughts['SPI_Values_Before'] = spi_before

df_droughts['Time_in_Drought_Before'] = pd.DataFrame(spi_before_num)


#########################################################################################
# How long in drought conditions after?
#########################################################################################
spi_after = []
spi_after_num = []
for i,(dates,polys) in tqdm(enumerate(zip(end_dates_droughts, polygons_droughts))):
	event_num = i+1
	spi_posterior = []
	event_length = len(np.where((df_droughts.Event_No == event_num))[0])
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
	
df_droughts['SPI_Values_After'] = np.nan
df_droughts['SPI_Values_After'] = df_droughts['SPI_Values_After'].astype('object')
df_droughts['SPI_Values_After'] = spi_after

df_droughts['Time_in_Drought_After'] = pd.DataFrame(spi_after_num)

#########################################################################################
## Total Time 
#########################################################################################
df_droughts['Total_Time'] = '' 

for i in tqdm(range(0,np.nanmax(df_droughts.Event_No))):
	subset = df_droughts.iloc[np.where((df_droughts.Event_No == i+1))[0]]
	df_droughts.loc[subset.index,'Total_Time'] = subset.Time_in_Drought_Before+subset.Time_in_Drought_After+len(subset)

#########################################################################################
# Save out file
#########################################################################################
df_droughts.to_csv(f'/data2/bpuxley/Events/independent_events_droughts.csv', index=False)

#########################################################################################
# Pluvial Events
#########################################################################################
#########################################################################################
# How long in pluvial conditions prior?
#########################################################################################
spi_before = []
spi_before_num = []
for i,(dates,polys) in tqdm(enumerate(zip(start_dates_pluvials, polygons_pluvials))):
	event_num = i+1
	spi_prior = []
	event_length = len(np.where((df_pluvials.Event_No == event_num))[0])
	for d in range(0, 1000):
		analysis_date = (pd.to_datetime(dates) + timedelta(days=-d)).strftime('%Y-%m-%d')
		avg_spi = calc_spi_avg(lons, lats, polys, spi.spi_30day.sel(time=analysis_date))
	
		if avg_spi >= 0:
			spi_prior.append(avg_spi)
		else:
			break
	
	spi_before_num += ([len(spi_prior)]*event_length)	
	spi_before += ([spi_prior]*event_length)
	
df_pluvials['SPI_Values_Before'] = np.nan
df_pluvials['SPI_Values_Before'] = df_pluvials['SPI_Values_Before'].astype('object')
df_pluvials['SPI_Values_Before'] = spi_before

df_pluvials['Time_in_Pluvial_Before'] = pd.DataFrame(spi_before_num)

#########################################################################################
# How long in pluvial condiitons after?
#########################################################################################
spi_after = []
spi_after_num = []
for i,(dates,polys) in tqdm(enumerate(zip(end_dates_pluvials, polygons_pluvials))):
	event_num = i+1
	spi_posterior = []
	event_length = len(np.where((df_pluvials.Event_No == event_num))[0])
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
	
df_pluvials['SPI_Values_After'] = np.nan
df_pluvials['SPI_Values_After'] = df_pluvials['SPI_Values_After'].astype('object')
df_pluvials['SPI_Values_After'] = spi_after

df_pluvials['Time_in_Pluvial_After'] = pd.DataFrame(spi_after_num)

#########################################################################################
## Total Time 
#########################################################################################
df_pluvials['Total_Time'] = '' 

for i in tqdm(range(0,np.nanmax(df_pluvials.Event_No))):
	subset = df_pluvials.iloc[np.where((df_pluvials.Event_No == i+1))[0]]
	df_pluvials.loc[subset.index,'Total_Time'] = subset.Time_in_Pluvial_Before+subset.Time_in_Pluvial_After+len(subset)



#########################################################################################
# Save out file
#########################################################################################
df_pluvials.to_csv(f'/data2/bpuxley/Events/independent_events_pluvials.csv', index=False)

