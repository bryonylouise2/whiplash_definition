#########################################################################################
## Plot the distribution of all the areas 
## Bryony Louise Puxley
## Last Edited: Wednesday, August 13th, 2025 
## Input: The CSV files of all potential events created in 8.Area_Calculation.py.
## Output: A PNG file of a 4-panel plot that includes a histogram of the frequency of 
## all areas, and a line plot of the percentiles relating to each area for both types
## of precipitation whiplash events.
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
import os

#########################################################################################
#Import Functions
#########################################################################################
import functions

#########################################################################################
#Import Data
#########################################################################################
dirname= '/data2/bpuxley/droughts_and_pluvials/Databases/'

years = ['1915_1924', '1925_1934', '1935_1944', '1945_1954', '1955_1964', '1965_1974', 
			'1975_1984', '1985_1994', '1995_2004', '2005_2014', '2015_2020']

drought_pathfiles = []
pluvial_pathfiles = []

for i in years:
	drought_filename = 'potential_events_droughts_%s.csv'%(i)
	drought_pathfile = os.path.join(dirname, drought_filename)
	drought_pathfiles.append(drought_pathfile)
	
	pluvial_filename = 'potential_events_pluvials_%s.csv'%(i)
	pluvial_pathfile = os.path.join(dirname, pluvial_filename)
	pluvial_pathfiles.append(pluvial_pathfile)
	

df_droughts = pd.concat([pd.read_csv(f) for f in drought_pathfiles ], ignore_index=True)
df_pluvials = pd.concat([pd.read_csv(f) for f in pluvial_pathfiles ], ignore_index=True)

area_chosen =  175000

#########################################################################################
#Plot of area distributions
#########################################################################################
fig = plt.figure(figsize = (10,10), dpi = 300, tight_layout =True)

percs = np.arange(1,101,1)
bins = np.arange(0, 1600000, 25000)

# First subplot of the area distributions of drought-to-pluvial (histogram: Frequency vs Area)
ax1 = fig.add_subplot(221)
plt.hist(df_droughts.Area, bins, density = False, histtype ='bar',color='cornflowerblue', edgecolor ='k', linewidth=0.5)
plt.axvline(x=175000, color='r', linestyle ='--')

plt.xticks(np.arange(0, 2800000, 200000))
ax1.set_xticklabels(['0','','','','8,000,000','','','','1,600,000','','','','2,400,000',''], fontsize=10)
plt.yticks(np.arange(0, 65000, 5000))
ax1.set_yticklabels(['0','','10,000','','20,000','','30,000', '','40,000','','50,000','','60,000'], fontsize=10)

ax1.set_xlabel('Area ($km^2$)', fontsize = 10)
ax1.set_ylabel('Frequency', fontsize = 10)
plt.title("a) Distribution of Drought  KDE Polygon Areas", fontsize=12)

# Second subplot of the area distributions of drought-to-pluvial (Line: Area vs Percentile)
area_percs = np.nanpercentile(df_droughts.Area.tolist(), percs)

ax2 = fig.add_subplot(222)
plt.plot(percs, area_percs, color='k')
plt.scatter(scs.percentileofscore(area_percs, 175000), 175000, marker='o', color='r', s=25, zorder=2)

plt.xticks(np.arange(0,100,5))
ax2.set_xticklabels(['0','','10','','20','','30','','40','','50','','60','','70','','80','','90',''], fontsize=10)
plt.grid()
ax2.set_axisbelow(True)

plt.yticks(np.arange(0, 3000000, 200000))
ax2.set_yticklabels(['0','','4,000,000','','8,000,000','','1,000,000','','1,600,000','','2,000,000','','2,400,000','','2,800,000'], fontsize=10)
ax2.set_xlabel('Percentile', fontsize = 10)
ax2.set_ylabel('Area ($km^2$)', fontsize = 10)
plt.title("b) Drought Area Percentiles", fontsize=12)

# Third subplot of the area distributions of pluvial-to-drought (histogram: Frequency vs Area)
ax3 = fig.add_subplot(223)
plt.hist(df_pluvials.Area, bins, density = False, histtype ='bar',color='cornflowerblue', edgecolor ='k', linewidth=0.5)
plt.axvline(x=175000, color='r', linestyle ='--')

plt.xticks(np.arange(0, 2800000, 200000))
ax3.set_xticklabels(['0','','','','8,000,000','','','','1,600,000','','','','2,400,000',''], fontsize=10)
plt.yticks(np.arange(0, 65000, 5000))
ax3.set_yticklabels(['0','','10,000','','20,000','','30,000', '','40,000','','50,000','','60,000'], fontsize=10)

ax3.set_xlabel('Area ($km^2$)', fontsize = 10)
ax3.set_ylabel('Frequency', fontsize = 10)
plt.title("c) Distribution of Pluvial KDE Polygon Areas", fontsize=12)

# Fourth subplot of the area distributions of pluvial-to-drought (Line: Area vs Percentile)
area_percs = np.nanpercentile(df_pluvials.Area.tolist(), percs)

ax4 = fig.add_subplot(224)
plt.plot(percs, area_percs, color='k')
plt.scatter(scs.percentileofscore(area_percs, 175000), 175000, marker='o', color='r', s=25, zorder=2)

plt.xticks(np.arange(0,100,5))
ax4.set_xticklabels(['0','','10','','20','','30','','40','','50','','60','','70','','80','','90',''], fontsize=10)
plt.grid()
ax2.set_axisbelow(True)
plt.yticks(np.arange(0, 3000000, 200000))
ax4.set_yticklabels(['0','','4,000,000','','8,000,000','','1,000,000','','1,600,000','','2,000,000','','2,400,000','','2,800,000'], fontsize=10)
ax4.set_xlabel('Percentile', fontsize = 10)
ax4.set_ylabel('Area ($km^2$)', fontsize = 10)
plt.title("d) Pluvial Area Percentiles", fontsize=12)

plt.tight_layout()
#plt.show(block=False)

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/area_distributions_droughts_pluvials.png', bbox_inches = 'tight', pad_inches = 0.1)    
