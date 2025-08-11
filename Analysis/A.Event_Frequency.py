#########################################################################################
## A script to calculate the frequency of events throughout the timeframe, as well as 
## seasonally throughout the year/
## Bryony Louise Puxley
## Last Edited: Monday, August 11th, 2025 
## Input: Independent event files of Drought-to-Pluvial and Pluvial-to-Drought events. 
## ENSO (Oceanic Niño Index-ONI) data. 
## Output: PNG files. 1) Plot of the yearly ENSO - ONI Index from 1915 to 2020. 
## 2) The yearly frequency of the events; the average areal size; and the total area 
## impacted from 1915 to 2020 (individually). 3) the monthly frequency of the events, and 
## the monthly average areal size. 4) The trends on each half of the database. 5) The yearly 
## frequency of the events; the average areal size; and the total area impacted from 1915 to 
## 2020 (one plot).
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

years = np.arange(1915,2021,1)

#########################################################################################
# Import ENSO data - Oceanic Niño Index (ONI)
#########################################################################################
# greater than 0.5 (warm) = El Niño, less than -0.5 (cool) La Niña
#########################################################################################
#ONI
enso_oni =  pd.read_csv('/data2/bpuxley/ENSO/ENSO_ONI.csv')
#enso_oni['enso_phase'] = enso_oni['ONI'].apply(functions.categorize_enso_strength)

#Define ENSO Years
# Events are defined as five consecutive overlapping 3-month periods at or above the +0.5 anomaly 
#for warm (El Niño) events and at or below the -0.5 anomaly for cool (La Niña) events.

enso_oni['phase'] = 0
enso_oni['strength'] = 0

#categorize El Nino (1) vs La Nina (-1)
# Identify consecutive segments meeting the condition
threshold = 0.5
min_consecutive = 5

# Track the start and end indices of sequences
above_idx = enso_oni.index[enso_oni['ONI'] > threshold] #indices that are greater than the threshold
below_idx = enso_oni.index[enso_oni['ONI'] < -threshold] #indices that are less than the negative threshold

functions.categorize_enso_events(enso_oni, above_idx, 1, min_consecutive, threshold, col='phase')
functions.categorize_enso_events(enso_oni, below_idx, -1, min_consecutive, threshold, col='phase')

#categorize the strength of the El Niño event
min_consecutive = 3
strength_thresholds = [0.5, 1.0, 1.5, 2.0]
strength_values = {0.5: 1, 1.0: 2, 1.5: 3, 2.0: 4} #weak - 1, moderate - 2, strong - 3, very strong - 4

for threshold in strength_thresholds:
	print(f"\nProcessing threshold: {threshold}")
	 
	# Track the start and end indices of sequences
	above_idx = enso_oni.index[enso_oni['ONI'] > threshold] #indices that are greater than the threshold
	below_idx = enso_oni.index[enso_oni['ONI'] < -threshold] #indices that are less than the negative threshold

	functions.categorize_enso_events(enso_oni, above_idx, strength_values[threshold], min_consecutive, threshold, col='strength')
	functions.categorize_enso_events(enso_oni, below_idx, -strength_values[threshold], min_consecutive, -threshold, col='strength')

#Define ENSO years	
d = {'Years': years, 'phase': 0, 'strength': 0, 'avg_oni': 0}
enso_years = pd.DataFrame(d)

count=0
for i in tqdm(range(6,len(enso_oni),12)):
	subset = enso_oni[i:i+12]
	
	enso_years.loc[count, 'avg_oni'] = round(np.nanmean(subset.ONI),3)
	enso_years.loc[count,'phase'] = scs.mode(subset.phase).mode
	
	if scs.mode(subset.phase).mode == 0:
		enso_years.loc[count,'strength'] == 0
	elif scs.mode(subset.phase).mode < 0:
		enso_years.loc[count,'strength'] = np.nanmin(subset.strength)
	else:
		enso_years.loc[count,'strength'] = np.nanmax(subset.strength)
	count+=1

#########################################################################################
# Yearly and Seasonal Analysis
#########################################################################################	
#########################################################################################
# Calculate the frequency of the events throughout the time frame
#########################################################################################
#Extract the year from the start dates of the events and create a histogram
start_dates_dp = functions.start_dates(df_DP)
start_dates_pd = functions.start_dates(df_PD)

years_dp = pd.to_datetime(start_dates_dp).dt.year
years_pd = pd.to_datetime(start_dates_pd).dt.year

hist_dp_yearly = np.histogram(years_dp, bins=np.arange(1915,2022,1))
hist_pd_yearly = np.histogram(years_pd, bins=np.arange(1915,2022,1))

#########################################################################################
# Calculate the frequency of the events throughout the year
#########################################################################################
months_dp =  pd.to_datetime(start_dates_dp).dt.month
months_pd =  pd.to_datetime(start_dates_pd).dt.month

hist_dp_monthly = np.histogram(months_dp, bins=np.arange(1,14,1))
hist_pd_monthly = np.histogram(months_pd, bins=np.arange(1,14,1))

#########################################################################################
# Calculate how the areal size of events and the length of events are changing throughout time
#########################################################################################

def area_and_length(df):
	largest_area = []
	last_day = []
	for i in tqdm(range(0,np.nanmax(df.Event_No))):
		#subset database to individual events
		subset_ind = np.where((df.Event_No == i+1))[0] 
		subset =  df.iloc[subset_ind]
	
		largest_area.append(subset.loc[subset.Area.idxmax()]) #find the day of the event with the largest area
		last_day.append(subset.loc[subset.Day_No.idxmax()]) #find the last day of the event

	largest_area = pd.DataFrame(largest_area).reset_index(drop=True)
	last_day = pd.DataFrame(last_day).reset_index(drop=True)
	
	#Add one day to Day_No before averaging to help with interpretation
	#If last_day.Day_No = 0 then the event lasted 1 day, similarly of the last_day.Day_No = 3 the event lasted 4 days etc.
	last_day.Day_No = last_day.Day_No+1	
	
	#Add a year-only column
	largest_area['year'] = pd.to_datetime(largest_area.Whiplash_Date).dt.year
	last_day['year'] = pd.to_datetime(last_day.Whiplash_Date).dt.year
	
	#Add a month-only column to the largest area dataframes
	largest_area['month'] = pd.to_datetime(largest_area.Whiplash_Date).dt.month
	last_day['month'] = pd.to_datetime(last_day.Whiplash_Date).dt.month
	
	#Group by year and average and sum
	avg_size = largest_area.groupby('year')['Area'].mean() #average size of events
	total_area = largest_area.groupby('year')['Area'].sum() #total yearly area of events
	avg_length = last_day.groupby('year')['Day_No'].mean() #average length of events
	
	#Group by month and average
	avg_monthly_size = largest_area.groupby('month')['Area'].mean()
	avg_monthly_length = last_day.groupby('month')['Day_No'].mean()

	return avg_size, total_area, avg_length, avg_monthly_size, avg_monthly_length

avg_size_dp, total_area_dp, avg_length_dp, avg_monthly_size_dp, avg_monthly_length_dp = area_and_length(df_DP)
avg_size_pd, total_area_pd, avg_length_pd, avg_monthly_size_pd, avg_monthly_length_pd = area_and_length(df_PD)

#########################################################################################
# Calculate the quantile regression for all plots
#########################################################################################
df_dp = pd.DataFrame({'time': hist_dp_yearly[1][:-1], 'freq': hist_dp_yearly[0], 'avg_size':avg_size_dp, 'total_area':total_area_dp, 'avg_len':avg_length_dp})
df_pd = pd.DataFrame({'time': hist_pd_yearly[1][:-1], 'freq': hist_pd_yearly[0], 'avg_size':avg_size_pd, 'total_area':total_area_pd, 'avg_len':avg_length_pd})

categories = ['freq', 'avg_size', 'total_area', 'avg_len'] # Define categories
quantiles = [0.1, 0.5, 0.9] # Define quantiles

quantile_regression_dp = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(df_dp['time'], df_dp[cat], quantiles)] }
quantile_regression_pd = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(df_pd['time'], df_pd[cat], quantiles)] }

#########################################################################################
# Calculate the statistical significance using bias-corrected and accelerated (BCa) 
# bootstrapping (Efron 1987) with 10,000 replications at p < 0.05 and a null hypothesis of 
# no trend. 
#########################################################################################
niter = 10000 #Number of Iterations
confidence_level = 0.95 #confidence level
alpha = round(1 - confidence_level, 2)

stat_sig_dp = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4} for cat in categories for val1, val2, val3, val4 in [functions.bca_bootstrap(df_dp['time'], df_dp[cat], quantiles, niter, alpha)] }
stat_sig_pd = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4} for cat in categories for val1, val2, val3, val4 in [functions.bca_bootstrap(df_pd['time'], df_pd[cat], quantiles, niter, alpha)] }

#########################################################################################
# Correlation between ENSO years and events
#########################################################################################	
corrs_DP = {cat: {'statistics':round(val1,3), 'pvalue':round(val2,3)} for cat in categories for val1, val2 in [scs.pearsonr(enso_years.avg_oni, df_dp[cat])]}
corrs_PD = {cat: {'statistics':round(val1,3), 'pvalue':round(val2,3)} for cat in categories for val1, val2 in [scs.pearsonr(enso_years.avg_oni, df_pd[cat])]}

#########################################################################################
# mean and median values for 1915-2020
#########################################################################################
dp_stastics = {cat: {'mean':  round(np.nanmean(df_dp[cat]),2), 
					'median': round(np.nanmedian(df_dp[cat]),2), 
					'min': round(np.nanmin(df_dp[cat]),2),
					'max': round(np.nanmax(df_dp[cat]),2),
					'range':  round(np.nanmax(df_dp[cat])-np.nanmin(df_dp[cat]),2)} 
					
					for cat in categories}
					
pd_stastics = {cat: {'mean':  round(np.nanmean(df_pd[cat]),2), 
					'median': round(np.nanmedian(df_pd[cat]),2), 
					'min': round(np.nanmin(df_pd[cat]),2),
					'max': round(np.nanmax(df_pd[cat]),2),
					'range':  round(np.nanmax(df_pd[cat])-np.nanmin(df_pd[cat]),2)} 
					
					for cat in categories}

#########################################################################################
# Split Database into 2: (1915-1967)/(1968-2020)
#########################################################################################	
first_half_dp = df_dp[:53]
second_half_dp = df_dp[53:]

first_half_pd = df_pd[:53]
second_half_pd = df_pd[53:]

#Calculate the means of each period
means_dp = {cat: {'first_half': round(np.nanmean(first_half_dp[cat]),3), 'second_half': round(np.nanmean(second_half_dp[cat]),3)} for cat in categories}
means_pd = {cat: {'first_half': round(np.nanmean(first_half_pd[cat]),3), 'second_half': round(np.nanmean(second_half_pd[cat]),3)} for cat in categories}

#Permutation Test
permutation_results_dp = {cat: {'difference':val1, 'pvalue':val2} for cat in categories for val1, val2 in [functions.permutation_test(np.array(first_half_dp[cat]), np.array(second_half_dp[cat]), 10000)]}
permutation_results_pd = {cat: {'difference':val1, 'pvalue':val2} for cat in categories for val1, val2 in [functions.permutation_test(np.array(first_half_pd[cat]), np.array(second_half_pd[cat]), 10000)]}

#Quantile Regression on each separate half
quantile_regression_first_dp = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(first_half_dp['time'], first_half_dp[cat], quantiles)] }
quantile_regression_second_dp = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(second_half_dp['time'], second_half_dp[cat], quantiles)] }

quantile_regression_first_pd = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(first_half_pd['time'], first_half_pd[cat], quantiles)] }
quantile_regression_second_pd = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(second_half_pd['time'], second_half_pd[cat], quantiles)] }

#Significance Testing on each separate half
stat_sig_first_dp = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4} for cat in categories for val1, val2, val3, val4 in [functions.bca_bootstrap(first_half_dp['time'], first_half_dp[cat], quantiles, niter, alpha)] }
stat_sig_second_dp = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4} for cat in categories for val1, val2, val3, val4 in [functions.bca_bootstrap(second_half_dp['time'], second_half_dp[cat], quantiles, niter, alpha)] }

stat_sig_first_pd = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4} for cat in categories for val1, val2, val3, val4 in [functions.bca_bootstrap(first_half_pd['time'], first_half_pd[cat], quantiles, niter, alpha)] }
stat_sig_second_pd = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4} for cat in categories for val1, val2, val3, val4 in [functions.bca_bootstrap(second_half_pd['time'], second_half_pd[cat], quantiles, niter, alpha)] }


#########################################################################################
# Use a permutation test to see whether the months are significantly different from one another
#########################################################################################
df_dp_monthly = pd.DataFrame({'months': np.arange(1,13,1), 'freq': hist_dp_monthly[0], 'avg_size':avg_monthly_size_dp, 'avg_len':avg_monthly_length_dp})
df_pd_monthly = pd.DataFrame({'months': np.arange(1,13,1), 'freq': hist_pd_monthly[0], 'avg_size':avg_monthly_size_pd, 'avg_len':avg_monthly_length_pd})

categories = ['freq', 'avg_size', 'avg_len'] # Define categories
months = df_dp_monthly['months'].unique() #Define months - .unique() makes it into an array

# Store results separately for each category
for cat in categories:
    print(f"\nProcessing category: {cat}")

    # Convert data into a dictionary {month: values}
    monthly_data = {month: df_dp_monthly[df_dp_monthly['months'] == month][cat].dropna().values for month in months}

    # Store pairwise p-values
    pairwise_results_diff = {}
    pairwise_results_pvalue = {}

    for month1, month2 in itertools.combinations(months, 2):
        data1 = np.array(monthly_data[month1])
        data2 = np.array(monthly_data[month2])
        
        obs_diff, p_val = permutation_test(data1, data2)
        pairwise_results_diff[(month1, month2)] = obs_diff
        pairwise_results_pvalue[(month1, month2)] = p_val
        
    # Apply multiple testing correction (FDR correction)
    _, corrected_p_values, _, _ = smm.multipletests(list(pairwise_results_pvalue.values()), method="fdr_bh")
	
	# Store corrected p-values in a matrix format for the heatmap
    diff_matrix = pd.DataFrame(np.ones((len(months), len(months))), index=months, columns=months)
    p_value_matrix = pd.DataFrame(np.ones((len(months), len(months))), index=months, columns=months)

    # Fill in the matrices
    for (pair, (obs_diff, corrected_p)) in zip(pairwise_results_diff.keys(), zip(pairwise_results_diff.values(), corrected_p_values)):
        month1, month2 = pair
        diff_matrix.loc[month1, month2] = obs_diff
        diff_matrix.loc[month2, month1] = obs_diff  # Mirror for symmetry

        p_value_matrix.loc[month1, month2] = corrected_p
        p_value_matrix.loc[month2, month1] = corrected_p  # Mirror for symmetry
	
	# Generate a significance mask (asterisks for p-values)
    significance_mask = p_value_matrix.map(lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '')

    # Plot heatmap 
    plt.figure(figsize = (10,8), dpi = 300, tight_layout =True)
    ax = sns.heatmap(diff_matrix, annot=p_value_matrix, fmt='', cmap="coolwarm", linewidths=0.5, annot_kws={"size": 14}, cbar_kws={'label': 'Observed Difference'})

    plt.title(f"Pairwise Observed Differences with Significance for {cat}")

    plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/pairwise_permutation_test_monthly_%s.png'%(cat), bbox_inches = 'tight', pad_inches = 0.1)    



#########################################################################################
#Plot - ENSO
#########################################################################################
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

# Drought-to-Pluvial
ax1 = fig.add_subplot(111)

plt.plot(enso_oni.ONI, color='k', linewidth=1)
#plt.plot(enso_years.avg_oni, color='k', linewidth=1)

plt.axhline(0, color='k', linewidth=2)

plt.fill_between(enso_oni.index, enso_oni.ONI, where=(enso_oni.ONI > 0), color='lightcoral', alpha=0.3, label='El Niño')
plt.fill_between(enso_oni.index, enso_oni.ONI, where=(enso_oni.ONI < 0), color='cornflowerblue', alpha=0.3, label='La Niña')

#plt.fill_between(enso_years.index, enso_years.avg_oni, where=(enso_years.avg_oni > 0), color='lightcoral', alpha=0.3, label='El Niño')
#plt.fill_between(enso_years.index, enso_years.avg_oni, where=(enso_years.avg_oni < 0), color='cornflowerblue', alpha=0.3, label='La Niña')

plt.xticks(np.arange(0, 1272, 60))
ax1.set_xticklabels(['1915','1920','1925','1930','1935','1940','1945','1950','1955','1960','1965','1970','1975','1980','1985','1990','1995','2000','2005','2010','2015', '2020'], fontsize=7)
plt.yticks(np.arange(-3, 3.5, 0.5))
ax1.set_yticklabels(['-3.0','-2.5','-2.0','-1.5','-1.0','-0.5','0','0.5','1.0','1.5','2.0','2.5','3.0'], fontsize=7)

plt.grid()
plt.gca().set_axisbelow(True)

ax1.set_xlabel('Year', fontsize = 10)
ax1.set_ylabel('Oceanic Niño Index (ONI)', fontsize = 10)

plt.legend(loc='upper left', fontsize=7, framealpha=1)

plt.title('a) El Niño-Southern Oscillation (1915-2020)', loc ='left')

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/enso_avg_1915_2020.png', bbox_inches = 'tight', pad_inches = 0.1)    

#########################################################################################
#Plot - Yearly Analysis
#########################################################################################
categories = ['freq', 'avg_size', 'total_area', 'avg_len'] # Define categories
y_labels = {'freq': 'Number of Events', 'avg_size': 'Average Size of Event ($km^2$)', 'total_area':'Total Area Impacted ($km^2$)', 'avg_len':'Average Length of Event (Days)'}
y_ticks = {'freq': np.arange(0, 28, 2), 'avg_size': np.arange(0, 550000, 50000), 'total_area':np.arange(0, 8000000, 500000), 'avg_len':np.arange(0, 8, 1)}

for cat in categories:
	print(f"\nPlotting category: {cat}")
	
	fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)
	
	bins=np.arange(1915,2021,1)
	enso_cbar = [
    	'darkmagenta' if val == 4 else
    	'mediumvioletred' if val == 3 else
    	'palevioletred' if val == 2 else
    	'lightpink' if val == 1 else
    	'midnightblue' if val == -4 else
    	'royalblue' if val == -3 else
    	'cornflowerblue' if val == -2 else
    	'lightskyblue' if val == -1 else
   		'silver' 
    	for val in enso_years.strength
		]		
	# Drought-to-Pluvial
	ax1 = fig.add_subplot(211)
	
	plt.bar(bins, df_dp[cat], color = 'mediumpurple', alpha=1,  edgecolor ='k', width=1, linewidth=1)
	plt.plot(bins, quantile_regression_dp[cat]['predicts'][0.1], label=r'$\tau$ = 0.1, {0}, [{1}, {2}]'.format(quantile_regression_dp[cat]['slopes'][0.1], stat_sig_dp[cat]['lower_bounds'][0.1], stat_sig_dp[cat]['upper_bounds'][0.1]), linewidth=2, color='darkgreen')
	plt.plot(bins, quantile_regression_dp[cat]['predicts'][0.5], label=r'$\tau$ = 0.5, {0}, [{1}, {2}]'.format(quantile_regression_dp[cat]['slopes'][0.5], stat_sig_dp[cat]['lower_bounds'][0.5], stat_sig_dp[cat]['upper_bounds'][0.5]), linewidth=2, color='k')
	plt.plot(bins, quantile_regression_dp[cat]['predicts'][0.9], label=r'$\tau$ = 0.9, {0}, [{1}, {2}]'.format(quantile_regression_dp[cat]['slopes'][0.9], stat_sig_dp[cat]['lower_bounds'][0.9], stat_sig_dp[cat]['upper_bounds'][0.9]), linewidth=2, color='tab:blue')
	
	plt.xticks(np.arange(1915, 2021, 5))
	ax1.set_xticklabels(['1915','1920','1925','1930','1935','1940','1945','1950','1955','1960','1965','1970','1975','1980','1985','1990','1995','2000','2005','2010','2015', '2020'], fontsize=7)
	plt.yticks(y_ticks[cat])
	ax1.set_yticklabels(y_ticks[cat],fontsize=7)
	
	plt.grid()
	plt.gca().set_axisbelow(True)
	
	ax1.set_xlabel('Year', fontsize = 10)
	ax1.set_ylabel(y_labels[cat], fontsize = 10) #will change
	
	plt.legend(loc='upper left', fontsize=7, framealpha=1)
	
	plt.title('a) Drought-to-Pluvial', loc ='left')

	# Pluvial to Drought
	ax2 = fig.add_subplot(212)
	
	plt.bar(bins, df_pd[cat], color = 'lightcoral', alpha=1,  edgecolor ='k', width=1, linewidth=1)
	plt.plot(bins, quantile_regression_pd[cat]['predicts'][0.1], label=r'$\tau$ = 0.1, {0}, [{1}, {2}]'.format(quantile_regression_pd[cat]['slopes'][0.1], stat_sig_pd[cat]['lower_bounds'][0.1], stat_sig_pd[cat]['upper_bounds'][0.1]), linewidth=2, color='darkgreen')
	plt.plot(bins, quantile_regression_pd[cat]['predicts'][0.5], label=r'$\tau$ = 0.5, {0}, [{1}, {2}]'.format(quantile_regression_pd[cat]['slopes'][0.5], stat_sig_pd[cat]['lower_bounds'][0.5], stat_sig_pd[cat]['upper_bounds'][0.5]), linewidth=2, color='k')
	plt.plot(bins, quantile_regression_pd[cat]['predicts'][0.9], label=r'$\tau$ = 0.9, {0}, [{1}, {2}]'.format(quantile_regression_pd[cat]['slopes'][0.9], stat_sig_pd[cat]['lower_bounds'][0.9], stat_sig_pd[cat]['upper_bounds'][0.9]), linewidth=2, color='tab:blue')
	
	plt.xticks(np.arange(1915, 2021, 5))
	ax2.set_xticklabels(['1915','1920','1925','1930','1935','1940','1945','1950','1955','1960','1965','1970','1975','1980','1985','1990','1995','2000','2005','2010','2015', '2020'], fontsize=7)
	plt.yticks(y_ticks[cat])
	ax2.set_yticklabels(y_ticks[cat],fontsize=7)
	
	plt.grid()
	plt.gca().set_axisbelow(True)
	
	ax2.set_xlabel('Year', fontsize = 10)
	ax2.set_ylabel(y_labels[cat], fontsize = 10) #will change
	
	plt.legend(loc='upper left', fontsize=7, framealpha=1)
	
	plt.title('b) Pluvial-to-Drought', loc='left')
	
	plt.savefig(f'/home/bpuxley/Definition_and_Climatology/Plots/event_yearly_{cat}.png', bbox_inches = 'tight', pad_inches = 0.1)    


#########################################################################################
#Plot - Monthly Analysis
#########################################################################################
categories = ['freq', 'avg_size', 'avg_len'] # Define categories
y_labels = {'freq': 'Number of Events', 'avg_size': 'Average Size of Event ($km^2$)', 'avg_len':'Average Length of Event $\it{days}$'}
y_ticks = {'freq': np.arange(0, 160, 10), 'avg_size': np.arange(0, 400000, 50000), 'avg_len': np.arange(0, 6, 1)}

for cat in categories:
	print(f"\nPlotting category: {cat}")
	
	fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)
	
	bins=np.arange(1,13,1)
	enso_cbar = ['cornflowerblue' if val == -1 else 'lightpink' if val == 1 else 'silver' for val in enso_years.ENSO_phrase]
	# Drought-to-Pluvial
	ax1 = fig.add_subplot(211)
	
	plt.bar(bins, df_dp_monthly[cat], color ='mediumpurple', alpha=1,  edgecolor ='k', width=1, linewidth=1)
	
	plt.xticks(np.arange(1, 13, 1))
	ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov', 'Dec'], fontsize=7)
	plt.yticks(y_ticks[cat])
	ax1.set_yticklabels(y_ticks[cat],fontsize=7)
	
	plt.grid()
	plt.gca().set_axisbelow(True)
	
	ax1.set_xlabel('Month', fontsize = 10)
	ax1.set_ylabel(y_labels[cat], fontsize = 10) #will change
	
	plt.title('a) Drought-to-Pluvial', loc ='left')

	# Pluvial to Drought
	ax2 = fig.add_subplot(212)
	
	plt.bar(bins,df_pd_monthly[cat], color ='lightcoral', alpha=1,  edgecolor ='k', width=1, linewidth=1)

	plt.xticks(np.arange(1, 13, 1))
	ax2.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov', 'Dec'], fontsize=7)
	plt.yticks(y_ticks[cat])
	ax2.set_yticklabels(y_ticks[cat],fontsize=7)
	
	plt.grid()
	plt.gca().set_axisbelow(True)
	
	ax2.set_xlabel('Month', fontsize = 10)
	ax2.set_ylabel(y_labels[cat], fontsize = 10) #will change
	
	plt.title('b) Pluvial-to-Drought', loc='left')
	
	plt.savefig(f'/home/bpuxley/Definition_and_Climatology/Plots/event_monthly_{cat}.png', bbox_inches = 'tight', pad_inches = 0.1)    


#########################################################################################
#Plot - Yearly Frequency - both halves
#########################################################################################
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

bins=np.arange(1915,2021,1)
bins_first=np.arange(1915,1968,1)
bins_second=np.arange(1968,2021,1)
enso_cbar = ['cornflowerblue' if val == -1 else 'lightpink' if val == 1 else 'silver' for val in enso_years.ENSO_phrase]

# Drought-to-Pluvial
ax1 = fig.add_subplot(211)

plt.bar(bins,hist_dp_yearly[0], color ='mediumpurple', alpha=1,  edgecolor ='k', width=1, linewidth=1)
plt.plot(bins_first, quantile_regression_first_dp['freq']['predicts'][0.1], label=r'$\tau$ = 0.1, {0}, [{1}, {2}]'.format(quantile_regression_first_dp['freq']['slopes'][0.1], stat_sig_first_dp['freq']['lower_bounds'][0.1], stat_sig_first_dp['freq']['upper_bounds'][0.1]), linewidth=2, color='darkgreen')
plt.plot(bins_first, quantile_regression_first_dp['freq']['predicts'][0.5], label=r'$\tau$ = 0.5, {0}, [{1}, {2}]'.format(quantile_regression_first_dp['freq']['slopes'][0.5], stat_sig_first_dp['freq']['lower_bounds'][0.5], stat_sig_first_dp['freq']['upper_bounds'][0.5]), linewidth=2, color='k')
plt.plot(bins_first, quantile_regression_first_dp['freq']['predicts'][0.9], label=r'$\tau$ = 0.9, {0}, [{1}, {2}]'.format(quantile_regression_first_dp['freq']['slopes'][0.9], stat_sig_first_dp['freq']['lower_bounds'][0.9], stat_sig_first_dp['freq']['upper_bounds'][0.9]), linewidth=2, color='tab:blue')

plt.plot(bins_second, quantile_regression_second_dp['freq']['predicts'][0.1], label=r'$\tau$ = 0.1, {0}, [{1}, {2}]'.format(quantile_regression_second_dp['freq']['slopes'][0.1], stat_sig_second_dp['freq']['lower_bounds'][0.1], stat_sig_second_dp['freq']['upper_bounds'][0.1]), linewidth=2, color='darkgreen')
plt.plot(bins_second, quantile_regression_second_dp['freq']['predicts'][0.5], label=r'$\tau$ = 0.5, {0}, [{1}, {2}]'.format(quantile_regression_second_dp['freq']['slopes'][0.5], stat_sig_second_dp['freq']['lower_bounds'][0.5], stat_sig_second_dp['freq']['upper_bounds'][0.5]), linewidth=2, color='k')
plt.plot(bins_second, quantile_regression_second_dp['freq']['predicts'][0.9], label=r'$\tau$ = 0.9, {0}, [{1}, {2}]'.format(quantile_regression_second_dp['freq']['slopes'][0.9], stat_sig_second_dp['freq']['lower_bounds'][0.9], stat_sig_second_dp['freq']['upper_bounds'][0.9]), linewidth=2, color='tab:blue')


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
plt.plot(bins_first, quantile_regression_first_pd['freq']['predicts'][0.1], label=r'$\tau$ = 0.1, {0}, [{1}, {2}]'.format(quantile_regression_first_pd['freq']['slopes'][0.1], stat_sig_first_pd['freq']['lower_bounds'][0.1], stat_sig_first_pd['freq']['upper_bounds'][0.1]), linewidth=2, color='darkgreen')
plt.plot(bins_first, quantile_regression_first_pd['freq']['predicts'][0.5], label=r'$\tau$ = 0.5, {0}, [{1}, {2}]'.format(quantile_regression_first_pd['freq']['slopes'][0.5], stat_sig_first_pd['freq']['lower_bounds'][0.5], stat_sig_first_pd['freq']['upper_bounds'][0.5]), linewidth=2, color='k')
plt.plot(bins_first, quantile_regression_first_pd['freq']['predicts'][0.9], label=r'$\tau$ = 0.9, {0}, [{1}, {2}]'.format(quantile_regression_first_pd['freq']['slopes'][0.9], stat_sig_first_pd['freq']['lower_bounds'][0.9], stat_sig_first_pd['freq']['upper_bounds'][0.9]), linewidth=2, color='tab:blue')

plt.plot(bins_second, quantile_regression_second_pd['freq']['predicts'][0.1], label=r'$\tau$ = 0.1, {0}, [{1}, {2}]'.format(quantile_regression_second_pd['freq']['slopes'][0.1], stat_sig_second_pd['freq']['lower_bounds'][0.1], stat_sig_second_pd['freq']['upper_bounds'][0.1]), linewidth=2, color='darkgreen')
plt.plot(bins_second, quantile_regression_second_pd['freq']['predicts'][0.5], label=r'$\tau$ = 0.5, {0}, [{1}, {2}]'.format(quantile_regression_second_pd['freq']['slopes'][0.5], stat_sig_second_pd['freq']['lower_bounds'][0.5], stat_sig_second_pd['freq']['upper_bounds'][0.5]), linewidth=2, color='k')
plt.plot(bins_second, quantile_regression_second_pd['freq']['predicts'][0.9], label=r'$\tau$ = 0.9, {0}, [{1}, {2}]'.format(quantile_regression_second_pd['freq']['slopes'][0.9], stat_sig_second_pd['freq']['lower_bounds'][0.9], stat_sig_second_pd['freq']['upper_bounds'][0.9]), linewidth=2, color='tab:blue')

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

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_yearly_frequency_halves.png', bbox_inches = 'tight', pad_inches = 0.1)    


#########################################################################################
#Plot - Yearly Analysis (one plot)
#########################################################################################
fig = plt.figure(figsize = (10,12), dpi = 300, tight_layout =True)

categories = ['freq', 'avg_size', 'total_area', 'avg_len'] # Define categories
y_labels = {'freq': 'Number of Events', 'avg_size': 'Average Size of Event ($km^2$)', 'total_area':'Total Area Impacted ($km^2$)', 'avg_len':'Average Length of Event ($\it{days}$)'}
y_ticks = {'freq': np.arange(0, 28, 2), 'avg_size': np.arange(0, 550000, 50000), 'total_area':np.arange(0, 8000000, 500000), 'avg_len':np.arange(0, 8, 1)}

bins=np.arange(1915,2021,1)

#########################################################################################
#Number of Events (Frequency)
#########################################################################################
cat = 'freq'
	
# Drought-to-Pluvial
ax1 = fig.add_subplot(321)
	
plt.bar(bins, df_dp[cat], color = 'mediumpurple', alpha=1,  edgecolor ='k', width=1, linewidth=1)
plt.plot(bins, quantile_regression_dp[cat]['predicts'][0.1], label=r'$\tau$ = 0.1, {0}, [{1}, {2}]'.format(quantile_regression_dp[cat]['slopes'][0.1], stat_sig_dp[cat]['lower_bounds'][0.1], stat_sig_dp[cat]['upper_bounds'][0.1]), linewidth=2, color='darkgreen')
plt.plot(bins, quantile_regression_dp[cat]['predicts'][0.5], label=r'$\tau$ = 0.5, {0}, [{1}, {2}]'.format(quantile_regression_dp[cat]['slopes'][0.5], stat_sig_dp[cat]['lower_bounds'][0.5], stat_sig_dp[cat]['upper_bounds'][0.5]), linewidth=2, color='k')
plt.plot(bins, quantile_regression_dp[cat]['predicts'][0.9], label=r'$\tau$ = 0.9, {0}, [{1}, {2}]'.format(quantile_regression_dp[cat]['slopes'][0.9], stat_sig_dp[cat]['lower_bounds'][0.9], stat_sig_dp[cat]['upper_bounds'][0.9]), linewidth=2, color='tab:blue')
	
plt.xticks(np.arange(1915, 2021, 10))
ax1.set_xticklabels(['1915','1925','1935','1945','1955','1965','1975','1985','1995','2005','2015'], fontsize=7)
plt.yticks(y_ticks[cat])
ax1.set_yticklabels(y_ticks[cat],fontsize=7)
	
plt.grid()
plt.gca().set_axisbelow(True)
	
ax1.set_xlabel('Year', fontsize = 10)
ax1.set_ylabel(y_labels[cat], fontsize = 10) #will change
	
plt.legend(loc='upper left', fontsize=7, framealpha=1)
	
plt.title('a) Number of Events \nper Year', loc = "left",fontsize=10)
plt.title("Drought-to-Pluvial", loc = 'right',fontsize=10)


# Pluvial to Drought
ax2 = fig.add_subplot(322)
	
plt.bar(bins, df_pd[cat], color = 'lightcoral', alpha=1,  edgecolor ='k', width=1, linewidth=1)
plt.plot(bins, quantile_regression_pd[cat]['predicts'][0.1], label=r'$\tau$ = 0.1, {0}, [{1}, {2}]'.format(quantile_regression_pd[cat]['slopes'][0.1], stat_sig_pd[cat]['lower_bounds'][0.1], stat_sig_pd[cat]['upper_bounds'][0.1]), linewidth=2, color='darkgreen')
plt.plot(bins, quantile_regression_pd[cat]['predicts'][0.5], label=r'$\tau$ = 0.5, {0}, [{1}, {2}]'.format(quantile_regression_pd[cat]['slopes'][0.5], stat_sig_pd[cat]['lower_bounds'][0.5], stat_sig_pd[cat]['upper_bounds'][0.5]), linewidth=2, color='k')
plt.plot(bins, quantile_regression_pd[cat]['predicts'][0.9], label=r'$\tau$ = 0.9, {0}, [{1}, {2}]'.format(quantile_regression_pd[cat]['slopes'][0.9], stat_sig_pd[cat]['lower_bounds'][0.9], stat_sig_pd[cat]['upper_bounds'][0.9]), linewidth=2, color='tab:blue')
	
plt.xticks(np.arange(1915, 2021, 10))
ax2.set_xticklabels(['1915','1925','1935','1945','1955','1965','1975','1985','1995','2005','2015'], fontsize=7)
plt.yticks(y_ticks[cat])
ax2.set_yticklabels(y_ticks[cat],fontsize=7)
	
plt.grid()
plt.gca().set_axisbelow(True)
	
ax2.set_xlabel('Year', fontsize = 10)
ax2.set_ylabel(y_labels[cat], fontsize = 10) #will change
	
plt.legend(loc='upper left', fontsize=7, framealpha=1)

plt.title('b) Number of Events \nper Year', loc = "left",fontsize=10)
plt.title('Pluvial-to-Drought', loc='right',fontsize=10)
	
	
#########################################################################################
#Average Size of the Event
#########################################################################################
cat = 'avg_size'
	
# Drought-to-Pluvial
ax3 = fig.add_subplot(323)
	
plt.bar(bins, df_dp[cat], color = 'mediumpurple', alpha=1,  edgecolor ='k', width=1, linewidth=1)
plt.plot(bins, quantile_regression_dp[cat]['predicts'][0.1], label=r'$\tau$ = 0.1, {0}, [{1}, {2}]'.format(quantile_regression_dp[cat]['slopes'][0.1], stat_sig_dp[cat]['lower_bounds'][0.1], stat_sig_dp[cat]['upper_bounds'][0.1]), linewidth=2, color='darkgreen')
plt.plot(bins, quantile_regression_dp[cat]['predicts'][0.5], label=r'$\tau$ = 0.5, {0}, [{1}, {2}]'.format(quantile_regression_dp[cat]['slopes'][0.5], stat_sig_dp[cat]['lower_bounds'][0.5], stat_sig_dp[cat]['upper_bounds'][0.5]), linewidth=2, color='k')
plt.plot(bins, quantile_regression_dp[cat]['predicts'][0.9], label=r'$\tau$ = 0.9, {0}, [{1}, {2}]'.format(quantile_regression_dp[cat]['slopes'][0.9], stat_sig_dp[cat]['lower_bounds'][0.9], stat_sig_dp[cat]['upper_bounds'][0.9]), linewidth=2, color='tab:blue')
	
plt.xticks(np.arange(1915, 2021, 10))
ax3.set_xticklabels(['1915','1925','1935','1945','1955','1965','1975','1985','1995','2005','2015'], fontsize=7)
plt.yticks(y_ticks[cat])
ax3.set_yticklabels(y_ticks[cat],fontsize=7)
	
plt.grid()
plt.gca().set_axisbelow(True)
	
ax3.set_xlabel('Year', fontsize = 10)
ax3.set_ylabel(y_labels[cat], fontsize = 10) #will change
	
plt.legend(loc='upper left', fontsize=7, framealpha=1)
	
plt.title('c) Average Size of an \nEvent ($km^2$)', loc = "left",fontsize=10)
plt.title("Drought-to-Pluvial", loc = 'right',fontsize=10)


# Pluvial to Drought
ax4 = fig.add_subplot(324)
	
plt.bar(bins, df_pd[cat], color = 'lightcoral', alpha=1,  edgecolor ='k', width=1, linewidth=1)
plt.plot(bins, quantile_regression_pd[cat]['predicts'][0.1], label=r'$\tau$ = 0.1, {0}, [{1}, {2}]'.format(quantile_regression_pd[cat]['slopes'][0.1], stat_sig_pd[cat]['lower_bounds'][0.1], stat_sig_pd[cat]['upper_bounds'][0.1]), linewidth=2, color='darkgreen')
plt.plot(bins, quantile_regression_pd[cat]['predicts'][0.5], label=r'$\tau$ = 0.5, {0}, [{1}, {2}]'.format(quantile_regression_pd[cat]['slopes'][0.5], stat_sig_pd[cat]['lower_bounds'][0.5], stat_sig_pd[cat]['upper_bounds'][0.5]), linewidth=2, color='k')
plt.plot(bins, quantile_regression_pd[cat]['predicts'][0.9], label=r'$\tau$ = 0.9, {0}, [{1}, {2}]'.format(quantile_regression_pd[cat]['slopes'][0.9], stat_sig_pd[cat]['lower_bounds'][0.9], stat_sig_pd[cat]['upper_bounds'][0.9]), linewidth=2, color='tab:blue')
	
plt.xticks(np.arange(1915, 2021, 10))
ax4.set_xticklabels(['1915','1925','1935','1945','1955','1965','1975','1985','1995','2005','2015'], fontsize=7)
plt.yticks(y_ticks[cat])
ax4.set_yticklabels(y_ticks[cat],fontsize=7)
	
plt.grid()
plt.gca().set_axisbelow(True)
	
ax4.set_xlabel('Year', fontsize = 10)
ax4.set_ylabel(y_labels[cat], fontsize = 10) #will change
	
plt.legend(loc='upper left', fontsize=7, framealpha=1)

plt.title('d) Average Size of an \nEvent ($km^2$)', loc = "left",fontsize=10)
plt.title('Pluvial-to-Drought', loc='right',fontsize=10)
	

#########################################################################################
#Total Area Impacted
#########################################################################################
cat = 'total_area'
	
# Drought-to-Pluvial
ax5 = fig.add_subplot(325)
	
plt.bar(bins, df_dp[cat], color = 'mediumpurple', alpha=1,  edgecolor ='k', width=1, linewidth=1)
plt.plot(bins, quantile_regression_dp[cat]['predicts'][0.1], label=r'$\tau$ = 0.1, {0}, [{1}, {2}]'.format(quantile_regression_dp[cat]['slopes'][0.1], stat_sig_dp[cat]['lower_bounds'][0.1], stat_sig_dp[cat]['upper_bounds'][0.1]), linewidth=2, color='darkgreen')
plt.plot(bins, quantile_regression_dp[cat]['predicts'][0.5], label=r'$\tau$ = 0.5, {0}, [{1}, {2}]'.format(quantile_regression_dp[cat]['slopes'][0.5], stat_sig_dp[cat]['lower_bounds'][0.5], stat_sig_dp[cat]['upper_bounds'][0.5]), linewidth=2, color='k')
plt.plot(bins, quantile_regression_dp[cat]['predicts'][0.9], label=r'$\tau$ = 0.9, {0}, [{1}, {2}]'.format(quantile_regression_dp[cat]['slopes'][0.9], stat_sig_dp[cat]['lower_bounds'][0.9], stat_sig_dp[cat]['upper_bounds'][0.9]), linewidth=2, color='tab:blue')
	
plt.xticks(np.arange(1915, 2021, 10))
ax5.set_xticklabels(['1915','1925','1935','1945','1955','1965','1975','1985','1995','2005','2015'], fontsize=7)
plt.yticks(y_ticks[cat])
ax5.set_yticklabels(y_ticks[cat],fontsize=7)
	
plt.grid()
plt.gca().set_axisbelow(True)
	
ax5.set_xlabel('Year', fontsize = 10)
ax5.set_ylabel(y_labels[cat], fontsize = 10) #will change
	
plt.legend(loc='upper left', fontsize=7, framealpha=1)
	
plt.title('e) Total Area Impacted \nper Year ($km^2$)', loc = "left",fontsize=10)
plt.title("Drought-to-Pluvial", loc = 'right',fontsize=10)


# Pluvial to Drought
ax6 = fig.add_subplot(326)
	
plt.bar(bins, df_pd[cat], color = 'lightcoral', alpha=1,  edgecolor ='k', width=1, linewidth=1)
plt.plot(bins, quantile_regression_pd[cat]['predicts'][0.1], label=r'$\tau$ = 0.1, {0}, [{1}, {2}]'.format(quantile_regression_pd[cat]['slopes'][0.1], stat_sig_pd[cat]['lower_bounds'][0.1], stat_sig_pd[cat]['upper_bounds'][0.1]), linewidth=2, color='darkgreen')
plt.plot(bins, quantile_regression_pd[cat]['predicts'][0.5], label=r'$\tau$ = 0.5, {0}, [{1}, {2}]'.format(quantile_regression_pd[cat]['slopes'][0.5], stat_sig_pd[cat]['lower_bounds'][0.5], stat_sig_pd[cat]['upper_bounds'][0.5]), linewidth=2, color='k')
plt.plot(bins, quantile_regression_pd[cat]['predicts'][0.9], label=r'$\tau$ = 0.9, {0}, [{1}, {2}]'.format(quantile_regression_pd[cat]['slopes'][0.9], stat_sig_pd[cat]['lower_bounds'][0.9], stat_sig_pd[cat]['upper_bounds'][0.9]), linewidth=2, color='tab:blue')
	
plt.xticks(np.arange(1915, 2021, 10))
ax6.set_xticklabels(['1915','1925','1935','1945','1955','1965','1975','1985','1995','2005','2015'], fontsize=7)
plt.yticks(y_ticks[cat])
ax6.set_yticklabels(y_ticks[cat],fontsize=7)
	
plt.grid()
plt.gca().set_axisbelow(True)
	
ax6.set_xlabel('Year', fontsize = 10)
ax6.set_ylabel(y_labels[cat], fontsize = 10) #will change
	
plt.legend(loc='upper left', fontsize=7, framealpha=1)

plt.title('f) Total Area Impacted \nper Year ($km^2$)', loc = "left",fontsize=10)
plt.title("Pluvial-to-Drought", loc = 'right',fontsize=10)

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_yearly_3.png', bbox_inches = 'tight', pad_inches = 0.1)    
