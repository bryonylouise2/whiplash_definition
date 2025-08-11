#########################################################################################
## Basic Analysis of Event Databases including:
##### On what "day" does the largest area occur?
##### On what "day" does the largest SPI change area average occur?
##### On what "day" does the largest SPI change max occur?
## Bryony Louise Puxley
## Last Edited: Monday, August 11th, 2025 
## Input: 
## Output: A PNG file of on what "day" does the largest area, the largest area-averaged 
## and max SPI change.  
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
# Import Functions
#########################################################################################
import functions

#########################################################################################
# Import Events - load previously made event files
#########################################################################################
events_DP = pd.read_csv('/data2/bpuxley/Events/independent_events_DP.csv')
events_PD = pd.read_csv('/data2/bpuxley/Events/independent_events_PD.csv')

#########################################################################################
# Call either Drought-to-Pluvial events or Pluvial-to-Drought Events
#########################################################################################
save_name = 'events_DP'
df = events_DP.copy()

#########################################################################################
# Analysis
#########################################################################################
day_largest_area = []
day_largest_spi_area = []
day_largest_spi_max = []

no_of_events = np.nanmax(df.Event_No)

for i in tqdm(range(0, no_of_events)):
	subset_ind = np.where((df.Event_No == i+1))[0]
	subset =  df.iloc[subset_ind]
	
	day_largest_area.append(subset.Day_No[subset['Area'].idxmax()])
	day_largest_spi_area.append(subset.Day_No[subset['Avg_SPI_Change'].idxmax()])
	day_largest_spi_max.append(subset.Day_No[subset['Max_SPI_Change'].idxmax()])
	
#########################################################################################
# Histograms For Plotting
#########################################################################################
hist_area = np.histogram(day_largest_area, bins=np.arange(0,22,1))
hist_spi_avg = np.histogram(day_largest_spi_area, bins=np.arange(0,22,1))
hist_spi_max = np.histogram(day_largest_spi_max, bins=np.arange(0,22,1))

#########################################################################################
# Plot
#########################################################################################
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

# First subplot 
ax1 = fig.add_subplot(111)

bins = np.arange(0, 21, 1)
width=0.3

plt.bar(bins,hist_area[0], color ='cornflowerblue', alpha=1,  edgecolor ='k', width=0.3, linewidth=1, label = 'Largest Area')
plt.bar(bins+width,hist_spi_avg[0], color ='mediumpurple', alpha=1,  edgecolor ='k', width=0.3, linewidth=1, label = 'Largest SPI (Avg)')
plt.bar(bins+2*width,hist_spi_max[0], color ='burlywood', alpha=1,  edgecolor ='k', width=0.3, linewidth=1, label = 'Largest SPI (Max)')

plt.xticks(np.arange(0.3, 21.3, 1))
ax1.set_xticklabels(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19', '20'], fontsize=10)
plt.yticks(np.arange(0, 1000, 100))
#ax1.set_yticklabels(['0','','5,000','','10,000','','15,000', '','20,000','','25,000','','30,000'], fontsize=10)

ax1.set_xlabel('Day', fontsize = 10)
ax1.set_ylabel('Frequency', fontsize = 10)

plt.legend(loc = 'upper right', fontsize=12)

plt.title('a) Drought-to-Pluvial')

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/information_%s.png'%(save_name), bbox_inches = 'tight', pad_inches = 0.1)    



