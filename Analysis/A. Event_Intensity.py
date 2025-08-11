#########################################################################################
## A script to examine the intensity of whiplash events
## Bryony Louise Puxley
## Last Edited: Monday, August 11th 2025 
## Input:
## Output: 
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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from itertools import cycle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import spei as si
import pandas as pd
import scipy.stats as scs
import shapely.wkt
import os

#########################################################################################
# Import Functions
#########################################################################################
import functions

#########################################################################################
# Import Events - load previously made event files
#########################################################################################
events_DP = pd.read_csv('/data2/bpuxley/Events/independent_events_DP.csv')
events_PD = pd.read_csv('/data2/bpuxley/Events/independent_events_PD.csv')

df_DP = events_DP.copy()
df_PD = events_PD.copy()

no_of_events_DP = np.nanmax(df_DP.Event_No)
no_of_events_PD = np.nanmax(df_PD.Event_No)

years = np.arange(1915,2021,1)

#########################################################################################
# Calculate the area-averaged SPI change and the maximum grid point SPI change for each event
#########################################################################################

def intensity(df):
	max_spi_change = []
	avg_spi_change = []
	
	time_before = []
	time_after = []
	
	cluster_no = []

	for i in tqdm(range(0,np.nanmax(df.Event_No))):
		#subset database to individual events
		subset_ind = np.where((df.Event_No == i+1))[0] 
		subset =  df.iloc[subset_ind]
	
		max_spi_change.append(subset.loc[subset.Max_SPI_Change.idxmax()]) #find the day of the event with the maximum grid point SPI change
		avg_spi_change.append(np.nanmean(subset.Avg_SPI_Change)) #find the average SPI change across all days and the whole area

		#find the time in condition before and after for later analysis - can be removed if not needed
		time_before.append(np.nanmean(subset[subset.columns[-4]]))
		time_after.append(np.nanmean(subset[subset.columns[-2]]))
		
		#find the cluster no
		cluster_no.append(np.nanmean(subset.cluster_no))
		
	max_spi_change = pd.DataFrame(max_spi_change).reset_index(drop=True)
	avg_spi_change = pd.DataFrame({'Event_No':  np.arange(0,np.nanmax(df.Event_No),1), 'Avg_SPI_Change': avg_spi_change, 'Max_SPI_Change': max_spi_change.Max_SPI_Change, 
									'Day_Max_Occurs': max_spi_change.Day_No, 'Whiplash_Date': max_spi_change.Whiplash_Date, 'Time_Before': time_before, 'Time_After':time_after, 'Cluster_No':cluster_no}).reset_index(drop=True)
	
	return avg_spi_change
	
intensity_dp = intensity(df_DP)
intensity_pd = intensity(df_PD)

#########################################################################################
# Add month and season columns
#########################################################################################
intensity_dp['month'] = pd.to_datetime(intensity_dp.Whiplash_Date).dt.month
intensity_pd['month'] = pd.to_datetime(intensity_pd.Whiplash_Date).dt.month

def categorize_season(value):
	if value == 12 or value == 1 or value == 2:
		return 1
	if value == 3 or value == 4 or value == 5:
		return 2
	if value == 6 or value == 7 or value == 8:
		return 3
	else:
		return 4
		
intensity_dp['season'] = intensity_dp['month'].apply(categorize_season)
intensity_pd['season'] = intensity_pd['month'].apply(categorize_season)

#########################################################################################
# Calculate Seasonal Averages
#########################################################################################
seasons = {'winter':1, 'spring':2, 'summer':3, 'fall':4}

seasonal_averages_DP = {key: [round(np.nanmean(intensity_dp[intensity_dp.season == value].Avg_SPI_Change),3),round(np.nanmean(intensity_dp[intensity_dp.season == value].Max_SPI_Change),3)] for key, value in seasons.items()} 
seasonal_averages_PD = {key: [round(np.nanmean(intensity_pd[intensity_pd.season == value].Avg_SPI_Change),3),round(np.nanmean(intensity_pd[intensity_pd.season == value].Max_SPI_Change),3)] for key, value in seasons.items()} 

#########################################################################################
# Calculate Cluster Averages
#########################################################################################
clusters = {'Cluster_1':1,'Cluster_2':2,'Cluster_3':3,'Cluster_4':4,'Cluster_5':5,'Cluster_6':6,'Cluster_7':7}

cluster_averages_DP = {key: [round(np.nanmean(intensity_dp[intensity_dp.Cluster_No == value].Avg_SPI_Change),3),round(np.nanmean(intensity_dp[intensity_dp.Cluster_No == value].Max_SPI_Change),3)] for key, value in clusters.items()} 
cluster_averages_PD = {key: [round(np.nanmean(intensity_pd[intensity_pd.Cluster_No == value].Avg_SPI_Change),3),round(np.nanmean(intensity_pd[intensity_pd.Cluster_No == value].Max_SPI_Change),3)] for key, value in clusters.items()} 

#########################################################################################
# Calculate Intensity Averages
#########################################################################################
bins = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]  # Adjust as needed
bin_labels = [f"{bins[i]} - {bins[i+1]}" for i in range(len(bins)-1)]

intensity_dp['Avg_SPI_Change_bins'] = pd.cut(intensity_dp['Avg_SPI_Change'], bins=bins, right=False)
intensity_pd['Avg_SPI_Change_bins'] = pd.cut(intensity_pd['Avg_SPI_Change'], bins=bins, right=False)

intensity_averages_DP = pd.DataFrame({'Intensity_bins': bin_labels, 'Time_Before': intensity_dp.groupby('Avg_SPI_Change_bins')['Time_Before'].mean().reset_index(drop=True), 'Time_After': intensity_dp.groupby('Avg_SPI_Change_bins')['Time_After'].mean().reset_index(drop=True)})
intensity_averages_PD = pd.DataFrame({'Intensity_bins': bin_labels, 'Time_Before': intensity_pd.groupby('Avg_SPI_Change_bins')['Time_Before'].mean().reset_index(drop=True), 'Time_After': intensity_pd.groupby('Avg_SPI_Change_bins')['Time_After'].mean().reset_index(drop=True)})



#########################################################################################
# Plot a scatter plot of Avg SPI Change vs Max SPI Change
#########################################################################################
fig = plt.figure(figsize = (10,5), dpi = 300, tight_layout =True)
'''
#season
colors_DP = intensity_dp.season
colors_PD = intensity_pd.season
vmin=1
vmax=5
'''
#Cluster
colors_DP = intensity_dp.Cluster_No
colors_PD = intensity_pd.Cluster_No
vmin=1
vmax=8
'''
#Season
cmap = ListedColormap(["cornflowerblue","mediumseagreen","lightpink","sandybrown"]) #season
colors = cycle(["cornflowerblue","mediumseagreen","lightpink","sandybrown"]) #season
'''
#Cluster
cmap = ListedColormap(['red', 'green', 'purple', 'saddlebrown', 'blue', 'black', 'orange'])
colors = cycle(['red', 'green', 'purple', 'saddlebrown', 'blue', 'black', 'orange'])

#Drought-to-Pluvial
ax1 = fig.add_subplot(121)

plt.scatter(intensity_dp.Avg_SPI_Change, intensity_dp.Max_SPI_Change, s=1, c=colors_DP,alpha=0.8,cmap=cmap,vmin=vmin,vmax=vmax)

for key,value in cluster_averages_DP.items():
	plt.plot(value[0], value[1], '*', color=next(colors), markersize=13, markeredgecolor='k', label=key)

plt.xticks(np.arange(1, 5.5, 0.5))
#ax1.set_xticklabels(np.arange(0, 1100, 100), fontsize=7)
plt.yticks(np.arange(1, 15, 1))
#ax1.set_yticklabels(np.arange(0, 1100, 100), fontsize=7)

ax1.set_xlabel('Area Averaged SPI Change', fontsize = 10)
ax1.set_ylabel('Maxiumum SPI Change (grid point)', fontsize = 10)	

# Add colorbar with custom ticks
cbar = plt.colorbar(pad=0.05)
'''
#seasonal plots
cbar.set_ticks(np.arange(1.5,5.5,1))
cbar.set_ticklabels(['DJF (winter)','MAM (spring)',' JJA (summer)','SON (fall)'])
cbar.ax.set_title('Season',fontsize=7)
'''
#cluster plots
cbar.set_ticks(np.arange(1.5,8.5,1))
cbar.set_ticklabels(['Cluster 1','Cluster 2',' Cluster 3','Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
cbar.ax.set_title('Cluster',fontsize=7)

cbar.ax.tick_params(labelsize=7) 	

#plt.legend(loc='upper left', fontsize=7, framealpha=1)
plt.title("a) Drought-to-Pluvial", fontsize=12, loc='left')

#Pluvial-to-Drought
ax2 = fig.add_subplot(122)

plt.scatter(intensity_pd.Avg_SPI_Change, intensity_pd.Max_SPI_Change, s=1, c=colors_PD, alpha=0.8,cmap=cmap,vmin=vmin,vmax=vmax)

for key,value in cluster_averages_PD.items():
	plt.plot(value[0], value[1], '*', color=next(colors), markersize=13, markeredgecolor='k', label=key)


plt.xticks(np.arange(1, 5.5, 0.5))
#ax2.set_xticklabels(np.arange(0, 1100, 100), fontsize=7)
plt.yticks(np.arange(1, 15, 1))
#ax2.set_yticklabels(np.arange(0, 1100, 100), fontsize=7)

ax2.set_xlabel('Area Averaged SPI Change', fontsize = 10)
ax2.set_ylabel('Maxiumum SPI Change (grid point)', fontsize = 10)	

# Add colorbar with custom ticks
cbar = plt.colorbar(pad=0.05)
'''
#seasonal plots
cbar.set_ticks(np.arange(1.5,5.5,1))
cbar.set_ticklabels(['DJF (winter)','MAM (spring)',' JJA (summer)','SON (fall)'])
cbar.ax.set_title('Season',fontsize=7)
'''
#cluster plots
cbar.set_ticks(np.arange(1.5,8.5,1))
cbar.set_ticklabels(['Cluster 1','Cluster 2',' Cluster 3','Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
cbar.ax.set_title('Cluster',fontsize=7)

cbar.ax.tick_params(labelsize=7)

#plt.legend(loc='upper left', fontsize=7, framealpha=1) 	
plt.title("b) Pluvial-to-Drought", fontsize=12, loc='left')

plt.tight_layout()
#plt.show(block=False)

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_intensity_cluster.png', bbox_inches = 'tight', pad_inches = 0.1)    

