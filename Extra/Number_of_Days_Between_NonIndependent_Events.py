#########################################################################################
## Calculate and plot the days between dates within the database.
## Bryony Louise
## Last Edited: Wednesday, January 23rd, 2025
## Input:
## Output: 
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

#########################################################################################
#Import Events - load previously made event files
#########################################################################################
events_DP = pd.read_csv('/data2/bpuxley/Events/events_DP.csv')
events_PD = pd.read_csv('/data2/bpuxley/Events/events_PD.csv')

#########################################################################################
#Question: What is the average time between dates in database?
#########################################################################################
days_between_DP = []

for i in tqdm(range(0, len(events_DP.Drought_Date)-1)):
	date1 = datetime.strptime(events_DP.Drought_Date[i], "%Y-%m-%d")
	date2 = datetime.strptime(events_DP.Drought_Date[i+1], "%Y-%m-%d")
	
	diff = date2 - date1
	
	days_between_DP.append(diff.days)
	
days_between_PD = []

for i in tqdm(range(0, len(events_PD.Pluvial_Date)-1)):
	date1 = datetime.strptime(events_PD.Pluvial_Date[i], "%Y-%m-%d")
	date2 = datetime.strptime(events_PD.Pluvial_Date[i+1], "%Y-%m-%d")
	
	diff = date2 - date1
	
	days_between_PD.append(diff.days)
	
#########################################################################################
#Plot as a histogram
#########################################################################################
fig = plt.figure(figsize = (10,5), dpi = 300, tight_layout =True)

bins = np.arange(0, 50, 1)

#Drought-to-Pluvial
ax1 = fig.add_subplot(121)

plt.hist(days_between_DP, bins, density = False, histtype ='bar',color='cornflowerblue', edgecolor ='k', linewidth=0.5)
plt.xticks(np.arange(0.5, 55.5, 5))
ax1.set_xticklabels(['0','5','10', '15','20', '25', '30', '35', '40', '45','50'], fontsize=10)
plt.yticks(np.arange(0, 3750, 250))
ax1.set_yticklabels(['0','','500', '','1,000', '', '1,500', '', '2,000', '','2,500','','3,000','','3,500'], fontsize=10)

ax1.set_xlabel('Number of Days', fontsize = 10)
ax1.set_ylabel('Frequency', fontsize = 10)
plt.title("a) Number of Days between Drought-to-Pluvial 'Events'", fontsize=12)

#Pluvial-to-Drought
ax2 = fig.add_subplot(122)

plt.hist(days_between_PD, bins, density = False, histtype ='bar',color='mediumpurple', edgecolor ='k', linewidth=0.5)
plt.xticks(np.arange(0.5, 55.5, 5))
ax2.set_xticklabels(['0','5','10', '15','20', '25', '30', '35', '40', '45','50'], fontsize=10)
plt.yticks(np.arange(0, 3750, 250))
ax2.set_yticklabels(['0','','500', '','1,000', '', '1,500', '', '2,000', '','2,500','','3,000','','3,500'], fontsize=10)


ax2.set_xlabel('Number of Days', fontsize = 10)
ax2.set_ylabel('Frequency', fontsize = 10)
plt.title("b) Number of Days between Pluvial-to-Drought 'Events'", fontsize=12)

plt.tight_layout()
#plt.show(block=False)

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/days_between_distributions.png', bbox_inches = 'tight', pad_inches = 0.1)    
