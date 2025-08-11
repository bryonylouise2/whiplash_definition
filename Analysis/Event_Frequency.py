#########################################################################################
## A script to calculate the frequency of events throughout the timeframe
## as well as seasonally throughout the year
## Bryony Louise
## Last Edited: Monday, August 11th, 2025 
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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import spei as si
import pandas as pd
import scipy.stats as scs
import statsmodels.api as sm
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

time = np.arange(1915,2021,1)

#########################################################################################
#Yearly Analysis
#########################################################################################	
#########################################################################################
#Calculate the frequency of the events throughout the time frame
#########################################################################################
#Extract the year from the start dates of the events and create a histogram
start_dates_dp = functions.start_dates(df_DP)
start_dates_pd = functions.start_dates(df_PD)

years_dp = pd.to_datetime(start_dates_dp).dt.year
years_pd = pd.to_datetime(start_dates_pd).dt.year

hist_dp_yearly = np.histogram(years_dp, bins=np.arange(1915,2022,1))
hist_pd_yearly = np.histogram(years_pd, bins=np.arange(1915,2022,1))

#########################################################################################
#Calculate how the areal size of events is changing throughout the time frame
#########################################################################################
#Drought-to-Pluvial
largest_area_dp = [] #largest area for each event

for i in tqdm(range(0,no_of_events_DP)):
	subset_ind = np.where((df_DP.Event_No == i+1))[0]
	subset =  df_DP.iloc[subset_ind]
	
	largest_area_dp.append(subset.loc[subset.Area.idxmax()])

largest_area_dp = pd.DataFrame(largest_area_dp).reset_index(drop=True)

#Pluvial-to-Drought
largest_area_pd = [] #largest area for each event

for i in tqdm(range(0,no_of_events_PD)):
	subset_ind = np.where((df_PD.Event_No == i+1))[0]
	subset =  df_PD.iloc[subset_ind]
	
	largest_area_pd.append(subset.loc[subset.Area.idxmax()])

largest_area_pd = pd.DataFrame(largest_area_pd).reset_index(drop=True)

#Add a year only column
largest_area_dp['year'] = pd.to_datetime(largest_area_dp.Whiplash_Date).dt.year
largest_area_pd['year'] = pd.to_datetime(largest_area_pd.Whiplash_Date).dt.year

#Group by year and average and sum
avg_size_dp = largest_area_dp.groupby('year')['Area'].mean()
avg_size_pd = largest_area_pd.groupby('year')['Area'].mean()

total_size_dp = largest_area_dp.groupby('year')['Area'].sum()
total_size_pd = largest_area_pd.groupby('year')['Area'].sum()

#########################################################################################
#Calculate how the length of events is changing throughout the time frame
#########################################################################################
#Drought-to-Pluvial
last_day_dp = [] #last day in each event

for i in tqdm(range(0, no_of_events_DP)):
	subset_ind = np.where((df_DP.Event_No == i+1))[0]
	subset =  df_DP.iloc[subset_ind]
	
	last_day_dp.append(subset.loc[subset.Day_No.idxmax()])
	
last_day_dp = pd.DataFrame(last_day_dp).reset_index(drop=True)

#Pluvial-to-Drought	
last_day_pd = [] #last day in each event

for i in tqdm(range(0, no_of_events_PD)):
	subset_ind = np.where((df_PD.Event_No == i+1))[0]
	subset =  df_PD.iloc[subset_ind]
	
	last_day_pd.append(subset.loc[subset.Day_No.idxmax()])
	
last_day_pd = pd.DataFrame(last_day_pd).reset_index(drop=True)
	
#Add a year only column
last_day_dp['year'] = pd.to_datetime(last_day_dp.Whiplash_Date).dt.year
last_day_pd['year'] = pd.to_datetime(last_day_pd.Whiplash_Date).dt.year

#Add one day to Day_No before averaging to help with interpretation
#If last_day.Day_No = 0 then the event lasted 1 day, similarly of the last_day.Day_No = 3 the event lasted 4 days etc.
last_day_dp.Day_No = last_day_dp.Day_No+1
last_day_pd.Day_No = last_day_pd.Day_No+1

#Group by year and average
avg_length_dp = last_day_dp.groupby('year')['Day_No'].mean()
avg_length_pd = last_day_pd.groupby('year')['Day_No'].mean()

#########################################################################################
#Calculate the quantile regression for all plots
#########################################################################################
time = hist_dp_yearly[1] #time

df_dp = pd.DataFrame({'time': time[:-1], 'frequency': hist_dp_yearly[0], 'avg_size':avg_size_dp, 'total_size':total_size_dp, 'avg_length':avg_length_dp})
df_pd = pd.DataFrame({'time': time[:-1], 'frequency': hist_pd_yearly[0], 'avg_size':avg_size_pd, 'total_size':total_size_pd, 'avg_length':avg_length_pd})

quantiles = [0.1, 0.5, 0.9] # Define quantiles

#frequency
x_range, predictions_freq_dp, slopes_freq_dp = functions.QuantileRegression(quantiles, df_dp, df_dp['frequency'])
x_range, predictions_freq_pd, slopes_freq_pd = functions.QuantileRegression(quantiles, df_pd, df_pd['frequency'])

#Average Size
x_range, predictions_avg_size_dp, slopes_avg_size_dp = functions.QuantileRegression(quantiles, df_dp, df_dp['avg_size'])
x_range, predictions_avg_size_pd, slopes_avg_size_pd = functions.QuantileRegression(quantiles, df_pd, df_pd['avg_size'])

#Total Size
x_range, predictions_total_size_dp, slopes_total_size_dp = functions.QuantileRegression(quantiles, df_dp, df_dp['total_size'])
x_range, predictions_total_size_pd, slopes_total_size_pd = functions.QuantileRegression(quantiles, df_pd, df_pd['total_size'])

#Average Length
x_range, predictions_avg_length_dp, slopes_avg_length_dp = functions.QuantileRegression(quantiles, df_dp, df_dp['avg_length'])
x_range, predictions_avg_length_pd, slopes_avg_length_pd = functions.QuantileRegression(quantiles, df_pd, df_pd['avg_length'])


#########################################################################################
#Seasonal Analysis
#########################################################################################	
#########################################################################################
#Calculate the frequency of the events throughout the year
#########################################################################################
months_dp =  pd.to_datetime(start_dates_dp).dt.month
months_pd =  pd.to_datetime(start_dates_pd).dt.month

hist_dp_monthly = np.histogram(months_dp, bins=np.arange(1,14,1))
hist_pd_monthly = np.histogram(months_pd, bins=np.arange(1,14,1))

#########################################################################################
#Calculate the average areal event size throughout the year
#########################################################################################
#Drought-to-Pluvial
#Add a month only column to largest area dataframes
largest_area_dp['month'] = pd.to_datetime(largest_area_dp.Whiplash_Date).dt.month
largest_area_pd['month'] = pd.to_datetime(largest_area_pd.Whiplash_Date).dt.month

#Group by month and average
avg_monthly_size_dp = largest_area_dp.groupby('month')['Area'].mean()
avg_monthly_size_pd = largest_area_pd.groupby('month')['Area'].mean()

#########################################################################################
#Calculate how the length of events is changing throughout the year
#########################################################################################
#Drought-to-Pluvial
#Add a month only column to last day dataframes
last_day_dp['month'] = pd.to_datetime(last_day_dp.Whiplash_Date).dt.month
last_day_pd['month'] = pd.to_datetime(last_day_pd.Whiplash_Date).dt.month

#Group by month and average
avg_monthly_length_dp = last_day_dp.groupby('month')['Day_No'].mean()
avg_monthly_length_pd = last_day_pd.groupby('month')['Day_No'].mean()

#########################################################################################
#Plot - Yearly Frequency
#########################################################################################
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

bins=np.arange(1915,2021,1)

# Drought-to-Pluvial
ax1 = fig.add_subplot(211)

plt.bar(bins,hist_dp_yearly[0], color ='mediumpurple', alpha=1,  edgecolor ='k', width=1, linewidth=1)
plt.plot(x_range, predictions_freq_dp[0.1], label=r'$\tau$ = 0.1, %s'%(slopes_freq_dp[0.1]), linewidth=2, color='darkgreen')
plt.plot(x_range, predictions_freq_dp[0.5], label=r'$\tau$ = 0.5, %s'%(slopes_freq_dp[0.5]), linewidth=2, color='k')
plt.plot(x_range, predictions_freq_dp[0.9], label=r'$\tau$ = 0.9, %s'%(slopes_freq_dp[0.9]), linewidth=2, color='tab:blue')

plt.xticks(np.arange(1915, 2021, 5))
ax1.set_xticklabels(['1915','1920','1925','1930','1935','1940','1945','1950','1955','1960','1965','1970','1975','1980','1985','1990','1995','2000','2005','2010','2015', '2020'], fontsize=7)
plt.yticks(np.arange(0, 28, 2))
ax1.set_yticklabels(['0','2','4','6','8','10','12','14','16','18','20','22','24','26'], fontsize=7)

plt.grid()
plt.gca().set_axisbelow(True)

ax1.set_xlabel('Year', fontsize = 10)
ax1.set_ylabel('Number of Events', fontsize = 10)

plt.legend(loc='upper left', fontsize=7, framealpha=1)

plt.title('a) Drought-to-Pluvial', loc ='left')

# Pluvial to Drought
ax2 = fig.add_subplot(212)

plt.bar(bins,hist_pd_yearly[0], color ='lightcoral', alpha=1,  edgecolor ='k', width=1, linewidth=1)
plt.plot(x_range, predictions_freq_pd[0.1], label=r'$\tau$ = 0.1, %s'%(slopes_freq_pd[0.1]), linewidth=2, color='darkgreen')
plt.plot(x_range, predictions_freq_pd[0.5], label=r'$\tau$ = 0.5, %s'%(slopes_freq_pd[0.5]), linewidth=2, color='k')
plt.plot(x_range, predictions_freq_pd[0.9], label=r'$\tau$ = 0.9, %s'%(slopes_freq_pd[0.9]), linewidth=2, color='tab:blue')

plt.xticks(np.arange(1915, 2021, 5))
ax2.set_xticklabels(['1915','1920','1925','1930','1935','1940','1945','1950','1955','1960','1965','1970','1975','1980','1985','1990','1995','2000','2005','2010','2015', '2020'], fontsize=7)
plt.yticks(np.arange(0, 28, 2))
ax2.set_yticklabels(['0','2','4','6','8','10','12','14','16','18','20','22','24','26'], fontsize=7)

plt.grid()
plt.gca().set_axisbelow(True)

ax2.set_xlabel('Year', fontsize = 10)
ax2.set_ylabel('Number of Events', fontsize = 10)

plt.legend(loc='upper left', fontsize=7, framealpha=1)

plt.title('b) Pluvial-to-Drought', loc='left')

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_yearly_frequency.png', bbox_inches = 'tight', pad_inches = 0.1)    

#########################################################################################
#Plot - Average size of event
#########################################################################################
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

bins=np.arange(1915,2021,1)

# Drought-to-Pluvial
ax1 = fig.add_subplot(211)

plt.bar(bins,avg_size_dp, color ='mediumpurple', alpha=1,  edgecolor ='k', width=1, linewidth=1)
plt.plot(x_range, predictions_avg_size_dp[0.1], label=r'$\tau$ = 0.1, %s'%(slopes_avg_size_dp[0.1]), linewidth=2, color='darkgreen')
plt.plot(x_range, predictions_avg_size_dp[0.5], label=r'$\tau$ = 0.5, %s'%(slopes_avg_size_dp[0.5]), linewidth=2, color='k')
plt.plot(x_range, predictions_avg_size_dp[0.9], label=r'$\tau$ = 0.9, %s'%(slopes_avg_size_dp[0.9]), linewidth=2, color='tab:blue')

plt.xticks(np.arange(1915, 2021, 5))
ax1.set_xticklabels(['1915','1920','1925','1930','1935','1940','1945','1950','1955','1960','1965','1970','1975','1980','1985','1990','1995','2000','2005','2010','2015', '2020'], fontsize=7)
plt.yticks(np.arange(0, 550000, 50000))
ax1.set_yticklabels(['0','50,000','100,000','150,000','200,000','250,000','300,000','350,000','400,000','450,000','500,000'], fontsize=7)

plt.grid()
plt.gca().set_axisbelow(True)

ax1.set_xlabel('Year', fontsize = 10)
ax1.set_ylabel('Average Size of Event ($km^2$)', fontsize = 10)

plt.legend(loc='upper left', fontsize=7, framealpha=1)

plt.title('a) Drought-to-Pluvial', loc ='left')

# Pluvial to Drought
ax2 = fig.add_subplot(212)

plt.bar(bins,avg_size_pd, color ='lightcoral', alpha=1,  edgecolor ='k', width=1, linewidth=1)
plt.plot(x_range, predictions_avg_size_pd[0.1], label=r'$\tau$ = 0.1, %s'%(slopes_avg_size_pd[0.1]), linewidth=2, color='darkgreen')
plt.plot(x_range, predictions_avg_size_pd[0.5], label=r'$\tau$ = 0.5, %s'%(slopes_avg_size_pd[0.5]), linewidth=2, color='k')
plt.plot(x_range, predictions_avg_size_pd[0.9], label=r'$\tau$ = 0.9, %s'%(slopes_avg_size_pd[0.9]), linewidth=2, color='tab:blue')

plt.xticks(np.arange(1915, 2021, 5))
ax2.set_xticklabels(['1915','1920','1925','1930','1935','1940','1945','1950','1955','1960','1965','1970','1975','1980','1985','1990','1995','2000','2005','2010','2015', '2020'], fontsize=7)
plt.yticks(np.arange(0, 550000, 50000))
ax2.set_yticklabels(['0','50,000','100,000','150,000','200,000','250,000','300,000','350,000','400,000','450,000','500,000'], fontsize=7)

plt.grid()
plt.gca().set_axisbelow(True)

ax2.set_xlabel('Year', fontsize = 10)
ax2.set_ylabel('Average Size of Event ($km^2$)', fontsize = 10)

plt.legend(loc='upper left', fontsize=7, framealpha=1)

plt.title('b) Pluvial-to-Drought', loc='left')

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_yearly_size_avg.png', bbox_inches = 'tight', pad_inches = 0.1)    

#########################################################################################
#Plot - Total Area Impacted
#########################################################################################
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

bins=np.arange(1915,2021,1)

# Drought-to-Pluvial
ax1 = fig.add_subplot(211)

plt.bar(bins,total_size_dp, color ='mediumpurple', alpha=1,  edgecolor ='k', width=1, linewidth=1)
plt.plot(x_range, predictions_total_size_dp[0.1], label=r'$\tau$ = 0.1, %s'%(slopes_total_size_dp[0.1]), linewidth=2, color='darkgreen')
plt.plot(x_range, predictions_total_size_dp[0.5], label=r'$\tau$ = 0.5, %s'%(slopes_total_size_dp[0.5]), linewidth=2, color='k')
plt.plot(x_range, predictions_total_size_dp[0.9], label=r'$\tau$ = 0.9, %s'%(slopes_total_size_dp[0.9]), linewidth=2, color='tab:blue')

plt.xticks(np.arange(1915, 2021, 5))
ax1.set_xticklabels(['1915','1920','1925','1930','1935','1940','1945','1950','1955','1960','1965','1970','1975','1980','1985','1990','1995','2000','2005','2010','2015', '2020'], fontsize=7)
plt.yticks(np.arange(0, 8000000, 500000))
ax1.set_yticklabels(['0','500,000','1,000,000','1,500,000','2,000,000','2,500,000','3,000,000','3,500,000','4,000,000','4,500,000','5,000,000','5,500,000','6,000,000','6,500,000','7,000,000','7,500,000'], fontsize=7)

plt.grid()
plt.gca().set_axisbelow(True)

ax1.set_xlabel('Year', fontsize = 10)
ax1.set_ylabel('Total Area Impacted ($km^2$)', fontsize = 10)

plt.legend(loc='upper left', fontsize=7, framealpha=1)

plt.title('a) Drought-to-Pluvial', loc ='left')

# Pluvial to Drought
ax2 = fig.add_subplot(212)

plt.bar(bins,total_size_pd, color ='lightcoral', alpha=1,  edgecolor ='k', width=1, linewidth=1)
plt.plot(x_range, predictions_total_size_pd[0.1], label=r'$\tau$ = 0.1, %s'%(slopes_total_size_pd[0.1]), linewidth=2, color='darkgreen')
plt.plot(x_range, predictions_total_size_pd[0.5], label=r'$\tau$ = 0.5, %s'%(slopes_total_size_pd[0.5]), linewidth=2, color='k')
plt.plot(x_range, predictions_total_size_pd[0.9], label=r'$\tau$ = 0.9, %s'%(slopes_total_size_pd[0.9]), linewidth=2, color='tab:blue')

plt.xticks(np.arange(1915, 2021, 5))
ax2.set_xticklabels(['1915','1920','1925','1930','1935','1940','1945','1950','1955','1960','1965','1970','1975','1980','1985','1990','1995','2000','2005','2010','2015', '2020'], fontsize=7)
plt.yticks(np.arange(0, 8000000, 500000))
ax2.set_yticklabels(['0','500,000','1,000,000','1,500,000','2,000,000','2,500,000','3,000,000','3,500,000','4,000,000','4,500,000','5,000,000','5,500,000','6,000,000','6,500,000','7,000,000','7,500,000'], fontsize=7)

plt.grid()
plt.gca().set_axisbelow(True)

ax2.set_xlabel('Year', fontsize = 10)
ax2.set_ylabel('Total Area Impacted ($km^2$)', fontsize = 10)

plt.legend(loc='upper left', fontsize=7, framealpha=1)

plt.title('b) Pluvial-to-Drought', loc='left')

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_yearly_size_total.png', bbox_inches = 'tight', pad_inches = 0.1)    


#########################################################################################
#Plot - Average length of event
#########################################################################################
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

bins=np.arange(1915,2021,1)

# Drought-to-Pluvial
ax1 = fig.add_subplot(211)

plt.bar(bins,avg_length_dp, color ='mediumpurple', alpha=1,  edgecolor ='k', width=1, linewidth=1)
plt.plot(x_range, predictions_avg_length_dp[0.1], label=r'$\tau$ = 0.1, %s'%(slopes_avg_length_dp[0.1]), linewidth=2, color='darkgreen')
plt.plot(x_range, predictions_avg_length_dp[0.5], label=r'$\tau$ = 0.5, %s'%(slopes_avg_length_dp[0.5]), linewidth=2, color='k')
plt.plot(x_range, predictions_avg_length_dp[0.9], label=r'$\tau$ = 0.9, %s'%(slopes_avg_length_dp[0.9]), linewidth=2, color='tab:blue')

plt.xticks(np.arange(1915, 2021, 5))
ax1.set_xticklabels(['1915','1920','1925','1930','1935','1940','1945','1950','1955','1960','1965','1970','1975','1980','1985','1990','1995','2000','2005','2010','2015', '2020'], fontsize=7)
plt.yticks(np.arange(0, 8, 1))
ax1.set_yticklabels(['0','1','2','3','4','5','6','7'], fontsize=7)

plt.grid()
plt.gca().set_axisbelow(True)

ax1.set_xlabel('Year', fontsize = 10)
ax1.set_ylabel('Average Length of Event (Days)', fontsize = 10)

plt.legend(loc='upper left', fontsize=7, framealpha=1)

plt.title('a) Drought-to-Pluvial',loc='left')

# Pluvial to Drought
ax2 = fig.add_subplot(212)

plt.bar(bins,avg_length_pd, color ='lightcoral', alpha=1,  edgecolor ='k', width=1, linewidth=1)
plt.plot(x_range, predictions_avg_length_pd[0.1], label=r'$\tau$ = 0.1, %s'%(slopes_avg_length_pd[0.1]), linewidth=2, color='darkgreen')
plt.plot(x_range, predictions_avg_length_pd[0.5], label=r'$\tau$ = 0.5, %s'%(slopes_avg_length_pd[0.5]), linewidth=2, color='k')
plt.plot(x_range, predictions_avg_length_pd[0.9], label=r'$\tau$ = 0.9, %s'%(slopes_avg_length_pd[0.9]), linewidth=2, color='tab:blue')

plt.xticks(np.arange(1915, 2021, 5))
ax2.set_xticklabels(['1915','1920','1925','1930','1935','1940','1945','1950','1955','1960','1965','1970','1975','1980','1985','1990','1995','2000','2005','2010','2015', '2020'], fontsize=7)
plt.yticks(np.arange(0, 8, 1))
ax2.set_yticklabels(['0','1','2','3','4','5','6','7'], fontsize=7)

plt.grid()
plt.gca().set_axisbelow(True)

ax2.set_xlabel('Year', fontsize = 10)
ax2.set_ylabel('Average Length of Event (Days)', fontsize = 10)

plt.legend(loc='upper left', fontsize=7, framealpha=1)

plt.title('b) Pluvial-to-Drought', loc='left')

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_yearly_avg_length.png', bbox_inches = 'tight', pad_inches = 0.1)    


#########################################################################################
#Plot - Monthly Frequency
#########################################################################################
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

bins=np.arange(1,13,1)

# Drought-to-Pluvial
ax1 = fig.add_subplot(211)

plt.bar(bins,hist_DP_monthly[0], color ='mediumpurple', alpha=1,  edgecolor ='k', width=1, linewidth=1)

plt.xticks(np.arange(1, 13, 1))
ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov', 'Dec'], fontsize=7)
plt.yticks(np.arange(0, 160, 10))
ax1.set_yticklabels(['0','10','20','30','40','50','60','70','80','90','100','110','120','130','140','150'], fontsize=7)

plt.grid()
plt.gca().set_axisbelow(True)

ax1.set_xlabel('Month', fontsize = 10)
ax1.set_ylabel('Number of Events', fontsize = 10)

plt.title('a) Drought-to-Pluvial', loc ='left')

# Pluvial to Drought
ax2 = fig.add_subplot(212)

plt.bar(bins,hist_PD_monthly[0], color ='lightcoral', alpha=1,  edgecolor ='k', width=1, linewidth=1)

plt.xticks(np.arange(1, 13, 1))
ax2.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov', 'Dec'], fontsize=7)
plt.yticks(np.arange(0, 160, 10))
ax2.set_yticklabels(['0','10','20','30','40','50','60','70','80','90','100','110','120','130','140','150'], fontsize=7)
plt.grid()
plt.gca().set_axisbelow(True)

ax2.set_xlabel('Month', fontsize = 10)
ax2.set_ylabel('Number of Events', fontsize = 10)

plt.title('b) Pluvial-to-Drought', loc='left')

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_monthly_frequency.png', bbox_inches = 'tight', pad_inches = 0.1)    

#########################################################################################
#Plot - Average Monthly Size
#########################################################################################
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

bins=np.arange(1,13,1)

# Drought-to-Pluvial
ax1 = fig.add_subplot(211)

plt.bar(bins,avg_monthly_size_dp, color ='mediumpurple', alpha=1,  edgecolor ='k', width=1, linewidth=1)

plt.xticks(np.arange(1, 13, 1))
ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov', 'Dec'], fontsize=7)
plt.yticks(np.arange(0, 400000, 50000))
ax1.set_yticklabels(['0','50,000','100,000','150,000','200,000','250,000','300,000','350,000'], fontsize=7)

plt.grid()
plt.gca().set_axisbelow(True)

ax1.set_xlabel('Month', fontsize = 10)
ax1.set_ylabel('Average Size of Event ($km^2$)', fontsize = 10)

plt.title('a) Drought-to-Pluvial', loc ='left')

# Pluvial to Drought
ax2 = fig.add_subplot(212)

plt.bar(bins,avg_monthly_size_pd, color ='lightcoral', alpha=1,  edgecolor ='k', width=1, linewidth=1)

plt.xticks(np.arange(1, 13, 1))
ax2.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov', 'Dec'], fontsize=7)
plt.yticks(np.arange(0, 400000, 50000))
ax2.set_yticklabels(['0','50,000','100,000','150,000','200,000','250,000','300,000','350,000'], fontsize=7)

plt.grid()
plt.gca().set_axisbelow(True)

ax2.set_xlabel('Month', fontsize = 10)
ax2.set_ylabel('Average Size of Event ($km^2$)', fontsize = 10)

plt.title('b) Pluvial-to-Drought', loc='left')

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_monthly_size_avg.png', bbox_inches = 'tight', pad_inches = 0.1)    

#########################################################################################
#Plot - Average Monthly Length
#########################################################################################
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

bins=np.arange(1,13,1)

# Drought-to-Pluvial
ax1 = fig.add_subplot(211)

plt.bar(bins,avg_monthly_length_dp, color ='mediumpurple', alpha=1,  edgecolor ='k', width=1, linewidth=1)

plt.xticks(np.arange(1, 13, 1))
ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov', 'Dec'], fontsize=7)
plt.yticks(np.arange(0, 6, 1))
ax1.set_yticklabels(['0','1','2','3','4','5'], fontsize=7)

plt.grid()
plt.gca().set_axisbelow(True)

ax1.set_xlabel('Month', fontsize = 10)
ax1.set_ylabel('Average Size of Event ($km^2$)', fontsize = 10)

plt.title('a) Drought-to-Pluvial', loc ='left')

# Pluvial to Drought
ax2 = fig.add_subplot(212)

plt.bar(bins,avg_monthly_length_pd, color ='lightcoral', alpha=1,  edgecolor ='k', width=1, linewidth=1)

plt.xticks(np.arange(1, 13, 1))
ax2.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov', 'Dec'], fontsize=7)
plt.yticks(np.arange(0, 6, 1))
ax2.set_yticklabels(['0','1','2','3','4','5'], fontsize=7)

plt.grid()
plt.gca().set_axisbelow(True)

ax2.set_xlabel('Month', fontsize = 10)
ax2.set_ylabel('Average Size of Event ($km^2$)', fontsize = 10)

plt.title('b) Pluvial-to-Drought', loc='left')

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_monthly_length_avg.png', bbox_inches = 'tight', pad_inches = 0.1)    


