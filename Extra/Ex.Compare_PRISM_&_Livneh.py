#########################################################################################
## A script to answer Reviewer Question #1: Can you evaluate whether there are notable 
## differences in values between Livneh and PRISM using overlapping time periods? I'd 
## like a bit more background to confirm the assumption that there is no influence on the 
## whiplash calculations between these datasets.
## Bryony Louise Puxley
## Last Edited: Thursday, September 11th, 2025
## Input:  Daily precipitation data from Livneh and PRISM from 1981 to 2020.
## Output: Two PNG files. 1) a timeseries of a) 30-day rolling precipitation totals in mm 
## for PRISM (blue) and Livneh (red) between 1981 and 2010, and b) the difference 
## (PRISM - Livneh) in 30-day rolling precipitation totals in mm. and 2) a spatial plot
## of the average annual precipitation total in inches (1981-2020) for a) PRISM and b) Livneh.
#########################################################################################
# Import Required Modules
#########################################################################################
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#########################################################################################
# Import Data
#########################################################################################
year1 = 1981 
year2 = 2010
years = np.arange(year1,year2+1,1) #array of years

#Import Livneh
livneh_pathfile = [] #list of pathfiles to open
livneh_path = '/data/deluge/reanalysis/REANALYSIS/Livneh/prec.YYYY.nc'

for i in years: #loop through each year
	livneh_pathfile.append(livneh_path.replace("YYYY", str(i)))

df_livneh = xr.open_mfdataset(livneh_pathfile, combine='by_coords') 

#Import PRISM
prism_pathfile = [] #list of pathfiles to open
prism_path = '/data/deluge/reanalysis/REANALYSIS/PRISM/daily/prec.YYYY.nc'

for i in years: #loop through each year
	prism_pathfile.append(prism_path.replace("YYYY", str(i)))

df_prism = xr.open_mfdataset(prism_pathfile, combine='by_coords') 

#########################################################################################
#Time Series of PRISM and Livneh Data (1981-2020) (1 panel plot)
#########################################################################################
prism_ts = df_prism.prec.mean(dim=["lat", "lon"])
livneh_ts = df_livneh.prec.mean(dim=["lat", "lon"])

#Compute the 30-day rolling precipitation totals to make plot visualization easier
print('About to calculate 30-day rolling sum')
window = 30
prism_rolling = prism_ts.rolling(time=window).sum().values
livneh_rolling = livneh_ts.rolling(time=window).sum().values
print('Calculated 30-day rolling sum')

precip_diff = prism_rolling - livneh_rolling

mask = ~np.isnan(prism_rolling) & ~np.isnan(livneh_rolling)
precip_corr = np.corrcoef(prism_rolling[mask], livneh_rolling[mask])[0,1]

#########################################################################################
# Plot the timeseries
#########################################################################################
print('Figure: Started')	
fig = plt.figure(figsize = (15,7), dpi = 300, tight_layout =True)

x = df_prism.time

# First subplot of the 30-day rolling precipitation sums of PRISM and Livneh 
ax1 = fig.add_subplot(211)
plt.grid(True, zorder=1)
plt.plot(x, prism_rolling, color='cornflowerblue', zorder=2, label ='PRISM')
plt.plot(x, livneh_rolling, color='lightcoral', zorder=2, label='Livneh')

#plt.xticks()
#ax1.set_xticklabels(np.arange(1980, 2015, 5) fontsize=10)
plt.yticks(np.arange(0, 140, 10))
#ax1.set_yticklabels(['0','','5,000','','10,000','','15,000', '','20,000','','25,000','','30,000'], fontsize=10)

plt.legend(loc='upper left')

ax1.set_xlabel('Year', fontsize = 10)
ax1.set_ylabel('30-day rolling precipitation totals (mm)', fontsize = 10)

plt.title('a) Timeseries of PRISM and Livneh Precipitation \n 1981-2010')

# second subplot of the difference in totals
ax2 = fig.add_subplot(212)
plt.grid(True, zorder=1)
plt.plot(x, precip_diff, color='cornflowerblue', zorder=2, label ='difference')

#plt.xticks()
#ax1.set_xticklabels(np.arange(1980, 2015, 5) fontsize=10)
plt.yticks(np.arange(-10, 12, 2))
#ax1.set_yticklabels(['0','','5,000','','10,000','','15,000', '','20,000','','25,000','','30,000'], fontsize=10)

plt.legend(loc='upper left')

ax2.set_xlabel('Year', fontsize = 10)
ax2.set_ylabel('30-day rolling precipitation difference (mm)', fontsize = 10)

plt.title('b) Timeseries of the Difference between PRISM and Livneh Precipitation \n 1981-2010')

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/precipitation_timeseries.png', bbox_inches = 'tight', pad_inches = 0.1)    
print('Figure: Saved')


#########################################################################################
#Average Annual Precipitation of both PRISM and Livneh Data (2 panel plot)
#########################################################################################
# Step 1: annual sum at each (lat, lon)
annual_sum_prism = df_prism.prec.groupby("time.year").sum(dim="time")
annual_sum_livneh = df_livneh.prec.groupby("time.year").sum(dim="time")

# Step 2: mean across years
mean_annual_prism = annual_sum_prism.mean(dim="year").values
mean_annual_livneh = annual_sum_livneh.mean(dim="year").values

mean_annual_difference = mean_annual_prism - mean_annual_livneh

#########################################################################################
# Plot the annual average precip
#########################################################################################
print('Figure: Started')
#Calculate the average annual precipitation across the region
fig = plt.figure(figsize = (6,7), dpi = 300, tight_layout =True)

# first subplot of PRISM precipitation
lon, lat = np.meshgrid(df_prism.lon.values, df_prism.lat.values) #create a meshgrid of lat,lon values

ax1 = fig.add_subplot(211, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs = plt.contourf(lon, lat, mean_annual_prism/25.4, transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 70, 2), cmap = 'terrain_r') 
cb1 = fig.colorbar(cs, ax=ax1, orientation='horizontal', pad=0.05)
cb1.set_label('Average Annual Precipitation (inches)')

plt.title('a) PRISM', loc = 'left')

# second subplot of Livneh precipitation
lon, lat = np.meshgrid(df_livneh.lon.values, df_livneh.lat.values) #create a meshgrid of lat,lon values

ax2 = fig.add_subplot(212, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linewidth=1)
ax2.add_feature(cfeature.STATES, edgecolor='black')

ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

cs = plt.contourf(lon, lat, mean_annual_livneh/25.4, transform=ccrs.PlateCarree(), extend = "max", levels=np.arange(0, 70, 2), cmap = 'terrain_r') 
cb2 = fig.colorbar(cs, ax=ax2, orientation='horizontal', pad=0.05, )
cb2.set_label('Average Annual Precipitation (inches)')

plt.title('b) Livneh', loc = 'left')

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/precipitation_annual', bbox_inches = 'tight', pad_inches = 0.1)    
print('Figure: Saved')

