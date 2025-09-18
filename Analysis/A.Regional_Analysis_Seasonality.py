#########################################################################################
## A script to calculate the seasonality of events throughout the timeframe for individual
## clusters
## Bryony Louise Puxley
## Last Edited: Wednesday, May 28th
## Input: Independent event files of Drought-to-Pluvial and Pluvial-to-Drought events. El
## Nino Southern Oscillation Southern Oscillation Index (SOI). Daily Precipitation Data from
## 1915-2020.
## Output: Six PNG files. 1) A PNG of the yearly ENSO index from 1915 to 2020. 2) A PNG of
## the temporal trends of drought-to-pluvial and pluvial-to-drought precipitation whiplash 
## events across CONUS (Figure 7 in the manuscript). 3) Two PNGs of the Temporal trends in 
## the  50th percentile (Figure 8) and 90th percentiles (Supplementary Figure 5) of 
## drought-to-pluvial and pluvial-to-drought precipitation whiplash events for each of the 
## 7 clusters. 4) A PNG of the seasonal cycle of the frequency of drought-to-pluvial and 
## pluvial-to-drought precipitation whiplash events (Figure 9) and finally 5) A PNG of the
## seasonal cycle of average areal size of event polygons for drought-to-pluvial and 
## pluvial-to-drought precipitation whiplash (Figure 10).
#########################################################################################
#Import Required Modules
#########################################################################################
import os
import numpy as np
import pandas as pd
import scipy.stats as scs
import statsmodels.api as sm
import xarray as xr
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import shapely.wkt

#########################################################################################
#Import Functions
#########################################################################################
import functions

def area_and_length(df):
	years = np.arange(1915,2021,1)
	largest_area = []
	last_day = []
	for i in tqdm(range(0,np.nanmax(df.event_no_temp))):
		#subset database to individual events
		subset_ind = np.where((df.event_no_temp == i+1))[0] 
		subset =  df.iloc[subset_ind]
	
		largest_area.append(subset.loc[subset.Area.idxmax()]) #find the day of the event with the largest area
		last_day.append(subset.loc[subset.Day_No.idxmax()]) #find the last day of the event

	largest_area = pd.DataFrame(largest_area).reset_index(drop=True)
	last_day = pd.DataFrame(last_day).reset_index(drop=True)
	
	#Add one day to Day_No before averaging to help with interpretation
	#If last_day.Day_No = 0 then the event lasted 1 day, similarly of the last_day.Day_No = 3 the event lasted 4 days etc.
	last_day.Day_No = last_day.Day_No+1	
	
	#Add a year only column
	largest_area['year'] = pd.to_datetime(largest_area.Whiplash_Date).dt.year
	last_day['year'] = pd.to_datetime(last_day.Whiplash_Date).dt.year
	
	#Add a month only column to largest area dataframes
	largest_area['month'] = pd.to_datetime(largest_area.Whiplash_Date).dt.month
	last_day['month'] = pd.to_datetime(last_day.Whiplash_Date).dt.month
	
	#Group by year and average and sum
	avg_size = largest_area.groupby('year')['Area'].mean() #average size of events
	total_area = largest_area.groupby('year')['Area'].sum() #total yearly area of events
	avg_length = last_day.groupby('year')['Day_No'].mean() #average length of events
	
	# Reindex to include all years, filling missing ones with 0
	avg_size = avg_size.reindex(years, fill_value=0)
	total_area = total_area.reindex(years, fill_value=0)
	avg_length = avg_length.reindex(years, fill_value=0)
	
	#Group by month and average
	avg_monthly_size = largest_area.groupby('month')['Area'].mean()
	avg_monthly_length = last_day.groupby('month')['Day_No'].mean()

	return avg_size, total_area, avg_length, avg_monthly_size, avg_monthly_length

def colorbar_limit_func(df, cat, q):
	fill_values = df[[str(cat)+'_dp_'+str(q), str(cat)+'_pd_'+str(q)]]
	min_value = fill_values.min().min()
	max_value = fill_values.max().max()
	colorbar_limit = abs(max([min_value, max_value], key=abs))
	
	return colorbar_limit

years = np.arange(1915,2021,1)
#########################################################################################
#Import ENSO data - Southern Oscillation Index (SOI)
#########################################################################################
# greater than 0.5 (warm) = El Nino, less than -0.5 (cool) La Nina
#########################################################################################
#ONI
enso_oni =  pd.read_csv('/data2/bpuxley/ENSO/ENSO_ONI.csv')
#enso_oni['enso_phase'] = enso_oni['ONI'].apply(functions.categorize_enso_strength)

#Define ENSO Years
# Events are defined as 5 consecutive overlapping 3-month periods at or above the +0.5 anomaly 
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

#categorize strength of el nino event
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
d = {'Years': years, 'phase': 0, 'strength':0, 'avg_oni':0}
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
# Import Events - load previously made event files
#########################################################################################
events_DP = pd.read_csv('/data2/bpuxley/Events/independent_events_DP.csv')
events_PD = pd.read_csv('/data2/bpuxley/Events/independent_events_PD.csv')

df_dp = events_DP.copy()
df_pd = events_PD.copy()

cluster_dp_databases = {'conus': df_dp, 
						'cluster_1': df_dp.iloc[np.where((df_dp.cluster_no == 1))].reset_index(drop=True), 
						'cluster_2': df_dp.iloc[np.where((df_dp.cluster_no == 2))].reset_index(drop=True),
						'cluster_3': df_dp.iloc[np.where((df_dp.cluster_no == 3))].reset_index(drop=True), 
						'cluster_4': df_dp.iloc[np.where((df_dp.cluster_no == 4))].reset_index(drop=True), 
						'cluster_5': df_dp.iloc[np.where((df_dp.cluster_no == 5))].reset_index(drop=True), 
						'cluster_6': df_dp.iloc[np.where((df_dp.cluster_no == 6))].reset_index(drop=True), 
						'cluster_7': df_dp.iloc[np.where((df_dp.cluster_no == 7))].reset_index(drop=True) 
						}
						
cluster_pd_databases = {'conus': df_pd, 
						'cluster_1': df_pd.iloc[np.where((df_pd.cluster_no == 1))].reset_index(drop=True), 
						'cluster_2': df_pd.iloc[np.where((df_pd.cluster_no == 2))].reset_index(drop=True),
						'cluster_3': df_pd.iloc[np.where((df_pd.cluster_no == 3))].reset_index(drop=True), 
						'cluster_4': df_pd.iloc[np.where((df_pd.cluster_no == 4))].reset_index(drop=True), 
						'cluster_5': df_pd.iloc[np.where((df_pd.cluster_no == 5))].reset_index(drop=True), 
						'cluster_6': df_pd.iloc[np.where((df_pd.cluster_no == 6))].reset_index(drop=True), 
						'cluster_7': df_pd.iloc[np.where((df_pd.cluster_no == 7))].reset_index(drop=True) 
						}						

no_of_events_dp = {'conus': len(cluster_dp_databases['conus'].iloc[np.where((cluster_dp_databases['conus'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_1': len(cluster_dp_databases['cluster_1'].iloc[np.where((cluster_dp_databases['cluster_1'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_2': len(cluster_dp_databases['cluster_2'].iloc[np.where((cluster_dp_databases['cluster_2'].Day_No == 0))].reset_index(drop=True)),
					'cluster_3': len(cluster_dp_databases['cluster_3'].iloc[np.where((cluster_dp_databases['cluster_3'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_4': len(cluster_dp_databases['cluster_4'].iloc[np.where((cluster_dp_databases['cluster_4'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_5': len(cluster_dp_databases['cluster_5'].iloc[np.where((cluster_dp_databases['cluster_5'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_6': len(cluster_dp_databases['cluster_6'].iloc[np.where((cluster_dp_databases['cluster_6'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_7': len(cluster_dp_databases['cluster_7'].iloc[np.where((cluster_dp_databases['cluster_7'].Day_No == 0))].reset_index(drop=True))
					 }

no_of_events_pd = {'conus': len(cluster_pd_databases['conus'].iloc[np.where((cluster_pd_databases['conus'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_1': len(cluster_pd_databases['cluster_1'].iloc[np.where((cluster_pd_databases['cluster_1'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_2': len(cluster_pd_databases['cluster_2'].iloc[np.where((cluster_pd_databases['cluster_2'].Day_No == 0))].reset_index(drop=True)),
					'cluster_3': len(cluster_pd_databases['cluster_3'].iloc[np.where((cluster_pd_databases['cluster_3'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_4': len(cluster_pd_databases['cluster_4'].iloc[np.where((cluster_pd_databases['cluster_4'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_5': len(cluster_pd_databases['cluster_5'].iloc[np.where((cluster_pd_databases['cluster_5'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_6': len(cluster_pd_databases['cluster_6'].iloc[np.where((cluster_pd_databases['cluster_6'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_7': len(cluster_pd_databases['cluster_7'].iloc[np.where((cluster_pd_databases['cluster_7'].Day_No == 0))].reset_index(drop=True))
					 }

#########################################################################################
#Calculate the frequency of the events throughout the time frame
#########################################################################################
clusters = ['conus','cluster_1','cluster_2','cluster_3','cluster_4','cluster_5','cluster_6','cluster_7']

#create empty dictionaries to store variables.
#yearly - frequency, avg_size, total_area, avg_len
df_dp_yearly_histograms = {}
df_pd_yearly_histograms = {}

#monthly - frequency, avg_size, avg_len
df_dp_monthly_histograms = {}
df_pd_monthly_histograms = {}

#quantile regression
quantile_regression_dp = {}
quantile_regression_pd = {}

#statistical significance
stat_sig_dp = {}
stat_sig_pd = {}

#correlation with enso
enso_corrs_dp = {}
enso_corrs_pd = {}

#means of each half
means_dp = {}
means_pd = {}

#permutation test results
permutation_results_dp = {}
permutation_results_pd = {}

#quantile regression
quantile_regression_first_half_dp = {}
quantile_regression_first_half_pd = {}

quantile_regression_second_half_dp = {}
quantile_regression_second_half_pd = {}

#statistical significance
stat_sig_first_half_dp = {}
stat_sig_first_half_pd = {}

stat_sig_second_half_dp = {}
stat_sig_second_half_pd = {}

for cluster in clusters:
	print(f"\nProcessing: {cluster}")
	
	df_dp = cluster_dp_databases[cluster]
	df_pd = cluster_pd_databases[cluster]
	
	#########################################################################################
	# Create a temp event number for analysis
	#########################################################################################
	day0_only_dp = df_dp.iloc[np.where((df_dp.Day_No == 0))].reset_index(drop=True)
	day0_only_pd = df_pd.iloc[np.where((df_pd.Day_No == 0))].reset_index(drop=True)
	
	mapping_dp = dict(zip(day0_only_dp['Event_No'], day0_only_dp.index+1))
	mapping_pd = dict(zip(day0_only_pd['Event_No'], day0_only_pd.index+1))

	df_dp['event_no_temp'] = [mapping_dp[val] for val in df_dp['Event_No']]
	df_pd['event_no_temp'] = [mapping_pd[val] for val in df_pd['Event_No']]
	
	#########################################################################################
	#Calculate the frequency of the events throughout the time frame
	#########################################################################################
	#Extract the year from the start dates of the events and create a histogram
	start_dates_dp = functions.start_dates(df_dp)
	start_dates_pd = functions.start_dates(df_pd)

	years_dp = pd.to_datetime(start_dates_dp).dt.year
	years_pd = pd.to_datetime(start_dates_pd).dt.year

	hist_yearly_freq_dp = np.histogram(years_dp, bins=np.arange(1915,2022,1))[0]
	hist_yearly_freq_pd = np.histogram(years_pd, bins=np.arange(1915,2022,1))[0]

	#########################################################################################
	#Calculate the frequency of the events throughout the year
	#########################################################################################
	months_dp =  pd.to_datetime(start_dates_dp).dt.month
	months_pd =  pd.to_datetime(start_dates_pd).dt.month

	hist_monthly_freq_dp = np.histogram(months_dp, bins=np.arange(1,14,1))[0]
	hist_monthly_freq_pd = np.histogram(months_pd, bins=np.arange(1,14,1))[0]
	
	#########################################################################################
	#Calculate how the areal size of events and length of events are changing throughout time
	#########################################################################################
	hist_yearly_avg_size_dp, hist_yearly_tot_area_dp, hist_yearly_avg_len_dp, hist_monthly_avg_size_dp, hist_monthly_avg_len_dp = area_and_length(df_dp)
	hist_yearly_avg_size_pd, hist_yearly_tot_area_pd, hist_yearly_avg_len_pd, hist_monthly_avg_size_pd, hist_monthly_avg_len_pd = area_and_length(df_pd)

	#########################################################################################
	#Assign to dictionaries
	#########################################################################################
	df_dp_yearly_histograms[cluster] = pd.DataFrame({'freq': hist_yearly_freq_dp, 'avg_size':hist_yearly_avg_size_dp, 'tot_area':hist_yearly_tot_area_dp, 'avg_len':hist_yearly_avg_len_dp})
	df_pd_yearly_histograms[cluster] = pd.DataFrame({'freq': hist_yearly_freq_pd, 'avg_size':hist_yearly_avg_size_pd, 'tot_area':hist_yearly_tot_area_pd, 'avg_len':hist_yearly_avg_len_pd})
 
	df_dp_monthly_histograms[cluster] = pd.DataFrame({'freq': hist_monthly_freq_dp, 'avg_size':hist_monthly_avg_size_dp, 'avg_len':hist_monthly_avg_len_dp})
	df_pd_monthly_histograms[cluster] = pd.DataFrame({'freq': hist_monthly_freq_pd, 'avg_size':hist_monthly_avg_size_pd, 'avg_len':hist_monthly_avg_len_pd})

	#########################################################################################
	#Calculate the quantile regression for all plots (yearly)
	#########################################################################################
	categories = ['freq', 'avg_size', 'tot_area', 'avg_len'] # Define categories
	quantiles = [0.1, 0.5, 0.9] # Define quantiles

	quantile_regression_dp[cluster] = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(df_dp_yearly_histograms[cluster].index, df_dp_yearly_histograms[cluster][cat], quantiles)] }
	quantile_regression_pd[cluster] = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(df_pd_yearly_histograms[cluster].index, df_pd_yearly_histograms[cluster][cat], quantiles)] }

	#########################################################################################
	#Calculate the statistical significance using using bias-corrected and accelerated (BCa) 
	#bootstrapping (Efron 1987) with 10 000 replications at p < 0.05 and a null hypothesis of 
	#no trend. 
	#########################################################################################
	niter = 10000 #Number of Iterations
	confidence_level = 0.95 #confidence level
	alpha = round(1 - confidence_level, 2)

	stat_sig_dp[cluster] = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4, 'sig':val5, 'linestyle':val6} for cat in categories for val1, val2, val3, val4, val5, val6 in [functions.bca_bootstrap(df_dp_yearly_histograms[cluster].index, df_dp_yearly_histograms[cluster][cat], quantiles, niter, alpha)] }
	stat_sig_pd[cluster] = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4, 'sig':val5, 'linestyle':val6} for cat in categories for val1, val2, val3, val4, val5, val6 in [functions.bca_bootstrap(df_pd_yearly_histograms[cluster].index, df_pd_yearly_histograms[cluster][cat], quantiles, niter, alpha)] }

	#########################################################################################
	#Correlation between ENSO years and events
	#########################################################################################	
	enso_corrs_dp[cluster] = {cat: {'statistics':round(val1,3), 'pvalue':round(val2,3)} for cat in categories for val1, val2 in [scs.pearsonr(enso_years.avg_oni, df_dp_yearly_histograms[cluster][cat])]}
	enso_corrs_pd[cluster] = {cat: {'statistics':round(val1,3), 'pvalue':round(val2,3)} for cat in categories for val1, val2 in [scs.pearsonr(enso_years.avg_oni, df_dp_yearly_histograms[cluster][cat])]}

	#########################################################################################
	#Split Database into 2: (1915-1967)/(1968-2020)
	#########################################################################################	
	first_half_dp = df_dp_yearly_histograms[cluster][:53]
	second_half_dp = df_dp_yearly_histograms[cluster][53:]

	first_half_pd = df_pd_yearly_histograms[cluster][:53]
	second_half_pd = df_pd_yearly_histograms[cluster][53:]

	#Calculate the means of each period
	means_dp[cluster] = {cat: {'first_half': round(np.nanmean(first_half_dp[cat]),3), 'second_half': round(np.nanmean(second_half_dp[cat]),3)} for cat in categories}
	means_pd[cluster] = {cat: {'first_half': round(np.nanmean(first_half_pd[cat]),3), 'second_half': round(np.nanmean(second_half_pd[cat]),3)} for cat in categories}

	#Permutation Test
	permutation_results_dp[cluster] = {cat: {'difference':val1, 'pvalue':val2} for cat in categories for val1, val2 in [functions.permutation_test(np.array(first_half_dp[cat]), np.array(second_half_dp[cat]), 10000)]}
	permutation_results_pd[cluster] = {cat: {'difference':val1, 'pvalue':val2} for cat in categories for val1, val2 in [functions.permutation_test(np.array(first_half_pd[cat]), np.array(second_half_pd[cat]), 10000)]}

	#Quantile Regression on each separate half
	quantile_regression_first_half_dp[cluster] = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(first_half_dp.index, first_half_dp[cat], quantiles)] }
	quantile_regression_second_half_dp[cluster] = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(second_half_dp.index, second_half_dp[cat], quantiles)] }

	quantile_regression_first_half_pd[cluster] = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(first_half_pd.index, first_half_pd[cat], quantiles)] }
	quantile_regression_second_half_pd[cluster] = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(second_half_pd.index, second_half_pd[cat], quantiles)] }

	#Significance Testing on each separate half
	stat_sig_first_half_dp[cluster] = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4} for cat in categories for val1, val2, val3, val4 in [functions.bca_bootstrap(first_half_dp.index, first_half_dp[cat], quantiles, niter, alpha)] }
	stat_sig_second_half_dp[cluster] = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4} for cat in categories for val1, val2, val3, val4 in [functions.bca_bootstrap(second_half_dp.index, second_half_dp[cat], quantiles, niter, alpha)] }

	stat_sig_first_half_pd[cluster] = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4} for cat in categories for val1, val2, val3, val4 in [functions.bca_bootstrap(first_half_pd.index, first_half_pd[cat], quantiles, niter, alpha)] }
	stat_sig_second_half_pd[cluster] = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4} for cat in categories for val1, val2, val3, val4 in [functions.bca_bootstrap(second_half_pd.index, second_half_pd[cat], quantiles, niter, alpha)] }


#########################################################################################
#Read in Cluster Data
#########################################################################################
clusters = ['cluster_1','cluster_2','cluster_3','cluster_4','cluster_5','cluster_6','cluster_7']
categories = ['freq', 'avg_size', 'tot_area', 'avg_len'] # Define categories
quantiles= [0.5,0.9]

cluster_polys = pd.read_csv('/data2/bpuxley/Events/cluster_polygons_final.csv')

cluster_polys['avg_poly'] = cluster_polys['avg_poly'].apply(shapely.wkt.loads)
cluster_polys = cluster_polys.drop('7', axis=1)

cluster_polys[['freq_dp_0.5', 
				'freq_dp_0.5_sig',
				'freq_dp_0.9', 
				'freq_dp_0.9_sig',
				'freq_pd_0.5', 
				'freq_pd_0.5_sig',
				'freq_pd_0.9', 
				'freq_pd_0.9_sig',
				
				'avg_size_dp_0.5', 
				'avg_size_dp_0.5_sig',
				'avg_size_dp_0.9', 
				'avg_size_dp_0.9_sig',
				'avg_size_pd_0.5', 
				'avg_size_pd_0.5_sig',
				'avg_size_pd_0.9', 
				'avg_size_pd_0.9_sig',
				
				'tot_area_dp_0.5', 
				'tot_area_dp_0.5_sig',
				'tot_area_dp_0.9', 
				'tot_area_dp_0.9_sig',
				'tot_area_pd_0.5', 
				'tot_area_pd_0.5_sig',
				'tot_area_pd_0.9', 
				'tot_area_pd_0.9_sig',
				
				'avg_len_dp_0.5', 
				'avg_len_dp_0.5_sig',
				'avg_len_dp_0.9', 
				'avg_len_dp_0.9_sig',
				'avg_len_pd_0.5', 
				'avg_len_pd_0.5_sig',
				'avg_len_pd_0.9', 
				'avg_len_pd_0.9_sig',
				]] = ''

#add columns to cluster_poly to represent trends and significance
for cat in categories:
	print(cat)
	for q in quantiles:
		print(q)
		i=0
		for cluster in clusters:
			#Slope
			cluster_polys.loc[i, str(cat)+'_dp_'+str(q)] = quantile_regression_dp[cluster][cat]['slopes'][q]
			cluster_polys.loc[i, str(cat)+'_pd_'+str(q)] = quantile_regression_pd[cluster][cat]['slopes'][q]
			
			#Significance
			cluster_polys.loc[i, str(cat)+'_dp_'+str(q)+'_sig'] = True if stat_sig_dp[cluster][cat]['pvals'][q] <= 0.05 else False
			cluster_polys.loc[i, str(cat)+'_pd_'+str(q)+'_sig'] = True if stat_sig_pd[cluster][cat]['pvals'][q] <= 0.05 else False
			
			i+=1

cluster_polys['freq_dp_0.5'] = [0.000, -0.011, 0.014, 0.000, 0.000, 0.000, 0.012]
cluster_polys['freq_dp_0.5_sig'] = [False, False, True, False, False, False, True]
cluster_polys['freq_dp_0.9'] = [0.000, 0.000, 0.038, 0.000, 0.000, 0.000, 0.000]
cluster_polys['freq_dp_0.9_sig'] = [False, False, True, False, False, False, False]

cluster_polys['freq_pd_0.5'] = [0.000, 0.000, 0.000, 0.024, -0.010, 0.000, 0.000]
cluster_polys['freq_pd_0.5_sig'] = [False, False, False, True, False, False, False]
cluster_polys['freq_pd_0.9'] = [0.010, 0.000, 0.017, 0.015, -0.024, 0.000, 0.000]
cluster_polys['freq_pd_0.9_sig'] = [True, False, True, True, False, False, False]

cluster_polys['avg_size_dp_0.5'] = [-71.548, 247.035, 453.104, -305.725, -385.731, 185.896, 2288.781]
cluster_polys['avg_size_dp_0.5_sig'] = [False, False, False, False, False, False, False]
cluster_polys['avg_size_dp_0.9'] = [-379.023, -68.089, 2140.367, -2359.699, -1077.588, 2598.674, 873.141]
cluster_polys['avg_size_dp_0.9_sig'] = [False, False, False, False, False, False, False]

cluster_polys['avg_size_pd_0.5'] = [350.590, 145.759, -6.815, 607.582, -218.932, -158.878, 0.000]
cluster_polys['avg_size_pd_0.5_sig'] = [False, False, False, False, False, False, False]
cluster_polys['avg_size_pd_0.9'] = [998.437, 786.217, 1004.822, 1589.423, -338.535, -131.135, -181.286]
cluster_polys['avg_size_pd_0.9_sig'] = [False, False, True, True, False, False, False]

cluster_polys['tot_area_dp_0.5'] = [-2285.683, -447.763, 4425.469, -436.902, 53.878, 1754.915, 2511.633]
cluster_polys['tot_area_dp_0.5_sig'] = [False, False, True, False, False, False, True]
cluster_polys['tot_area_dp_0.9'] = [-62.166, -1049.072, 8023.871, 1153.875, -4311.057, 6073.579, 2759.334]
cluster_polys['tot_area_dp_0.9_sig'] = [False, False, True, False, False, True, False]

cluster_polys['tot_area_pd_0.5'] = [1021.059, 699.248, 133.795, 5475.260, -1986.642, 2665.827, 0.000]
cluster_polys['tot_area_pd_0.5_sig'] = [False, False, False, True, False, False, False]
cluster_polys['tot_area_pd_0.9'] = [1972.683, -751.680, 5747.911, 5627.033, -7816.188, -7359.959, -2034.420]
cluster_polys['tot_area_pd_0.9_sig'] = [False, False, False, True, False, False, False]

cluster_polys['avg_len_dp_0.5'] = [-0.018, 0.016, -0.004, -0.005, -0.008, 0.000, 0.013]
cluster_polys['avg_len_dp_0.5_sig'] = [False, False, False, False, False, False, True]
cluster_polys['avg_len_dp_0.9'] = [0.000, 0.000, 0.027, 0.000, -0.038, -0.012, 0.029]
cluster_polys['avg_len_dp_0.9_sig'] = [False, False, False, False, False, False, False]

cluster_polys['avg_len_pd_0.5'] = [0.003, 0.004, -0.006, 0.021, -0.015, 0.013, 0.000]
cluster_polys['avg_len_pd_0.5_sig'] = [False, False, False, False, False, False, False]
cluster_polys['avg_len_pd_0.9'] = [0.023, 0.006, 0.003, 0.013, 0.000, -0.018, 0.015]
cluster_polys['avg_len_pd_0.9_sig'] = [False, False, False, False, False, False, False]


cluster_polys.to_csv('/data2/bpuxley/Events/temporal_trends.csv')

#########################################################################################
#########################################################################################
#Average Annual Rainfall
#########################################################################################
#########################################################################################
#Import Data
#########################################################################################
filename = 'prec.1915_2020.nc'
dirname= '/data2/bpuxley/Precip_Data'

pathfile = os.path.join(dirname, filename)
df_precip = xr.open_dataset(pathfile)

#Examine only the 30-year period from 1991-2020
df_precip = df_precip.sel(time=slice('1991-01-01', '2035-12-31'))

#Lons, lats
lons, lats = np.meshgrid(df_precip.lon.values, df_precip.lat.values) #create a meshgrid of lat,lon values
print('Read in Precipitation Data')

#Calculate the monthly climatology at each grid point
monthly_climatology = df_precip.groupby('time.month').mean(dim='time')

months=np.arange(1,13,1)

#Subset by each cluster and calculate the average monthly values for each cluster
monthly_rainfall = pd.DataFrame(columns=clusters, index=months)

##Deal with each cluster separately to save on memory.
for cluster in clusters:
	print(f"\nProcessing: {cluster}")
	
	#Get cluster polygon
	i = int(cluster[-1])
	polygon = cluster_polys['avg_poly'][i-1]
	
	# mask to only cluster region
	mask = functions._mask_outside_region(lons, lats, polygon)
	mask_3D = np.repeat(mask[np.newaxis,:,:], 12, axis=0)
	
	monthly_climatology_cluster =  np.ma.masked_array(monthly_climatology.prec, ~mask_3D)
	
	# mask any nans that slipped through
	monthly_climatology_cluster = np.ma.masked_invalid(monthly_climatology_cluster)  
	
	#Average across all the grid points
	monthly_avg = np.nanmean(np.nanmean(monthly_climatology_cluster, axis=1),axis=1)
	
	monthly_rainfall[cluster] = monthly_avg

#Convert to mm
monthly_rainfall = monthly_rainfall*25.4

monthly_rainfall['conus'] = monthly_rainfall.mean(axis=1)

#########################################################################################
#########################################################################################
#Plots
#########################################################################################
#########################################################################################
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
#Plot - Yearly Analysis (one plot - Figure 7)
#########################################################################################
cluster = 'conus'
categories = ['freq', 'avg_size', 'tot_area'] # Define categories
bins=np.arange(1915,2021,1)

y_labels = {'freq': 'Number of Events', 
			'avg_size': 'Average Size of Event ($km^2$)', 
			'tot_area':'Total Area Impacted ($km^2$)', 
			'avg_len':'Average Length of Event ($\it{days}$)'}

y_ticks = {'freq': np.arange(0, 28, 2), 
			'avg_size': np.arange(0, 650000, 50000), 
			'tot_area':np.arange(0, 9000000, 500000), 
			'avg_len':np.arange(0, 8, 1)}

plot_titles = {'freq': {'dp': 'a) Number of Events \nper Year', 'pd': 'b) Number of Events \nper Year'},
				'avg_size': {'dp':'c) Average Size of an \nEvent ($km^2$)', 'pd': 'd) Average Size of an \nEvent ($km^2$)'},
				'tot_area': {'dp': 'e) Total Area Impacted \nper Year ($km^2$)', 'pd':'f) Total Area Impacted \nper Year ($km^2$)'}, 
				'avg_len': {'dp': 'g) Average Number of Days \nAbove the Areal Threshold', 'pd': 'h) Average Number of Days \nAbove the Areal Threshold'}	
				}

fig = plt.figure(figsize = (10,12), dpi = 300, tight_layout =True)
i=1
for cat in categories:
	print(f"\nPlotting category: {cat}")
	
	# Drought-to-Pluvial
	print(i)
	ax1 = fig.add_subplot(4,2,i)
	
	plt.bar(bins, df_dp_yearly_histograms[cluster][cat], color = 'mediumpurple', alpha=1,  edgecolor ='k', width=1, linewidth=1)
	plt.plot(bins, quantile_regression_dp[cluster][cat]['predicts'][0.1], label=r'$\tau$ = 0.1, {0}'.format(quantile_regression_dp[cluster][cat]['slopes'][0.1]), linewidth=2, linestyle=stat_sig_dp[cluster][cat]['linestyle'][0.1], color='darkgreen')
	plt.plot(bins, quantile_regression_dp[cluster][cat]['predicts'][0.5], label=r'$\tau$ = 0.5, {0}'.format(quantile_regression_dp[cluster][cat]['slopes'][0.5]), linewidth=2, linestyle=stat_sig_dp[cluster][cat]['linestyle'][0.5], color='k')
	plt.plot(bins, quantile_regression_dp[cluster][cat]['predicts'][0.9], label=r'$\tau$ = 0.9, {0}'.format(quantile_regression_dp[cluster][cat]['slopes'][0.9]), linewidth=2, linestyle=stat_sig_dp[cluster][cat]['linestyle'][0.9], color='tab:blue')
	
	plt.xticks(np.arange(1915, 2021, 10))
	ax1.set_xticklabels(['1915','1925','1935','1945','1955','1965','1975','1985','1995','2005','2015'], fontsize=7)
	plt.yticks(y_ticks[cat])
	ax1.set_yticklabels(y_ticks[cat],fontsize=7)
	
	plt.grid()
	plt.gca().set_axisbelow(True)
		
	ax1.set_xlabel('Year', fontsize = 10)
	ax1.set_ylabel(y_labels[cat], fontsize = 10) #will change
	
	plt.legend(loc='upper left', fontsize=7, framealpha=1)
	
	plt.title(plot_titles[cat]['dp'], loc = "left",fontsize=10)
	plt.title("Drought-to-Pluvial", loc = 'right',fontsize=10)


	# Pluvial to Drought
	print(i+1)
	ax2 = fig.add_subplot(4,2,i+1)
	
	plt.bar(bins, df_pd_yearly_histograms[cluster][cat], color = 'lightcoral', alpha=1,  edgecolor ='k', width=1, linewidth=1)
	plt.plot(bins, quantile_regression_pd[cluster][cat]['predicts'][0.1], label=r'$\tau$ = 0.1, {0}'.format(quantile_regression_pd[cluster][cat]['slopes'][0.1]), linewidth=2, linestyle=stat_sig_pd[cluster][cat]['linestyle'][0.1], color='darkgreen')
	plt.plot(bins, quantile_regression_pd[cluster][cat]['predicts'][0.5], label=r'$\tau$ = 0.5, {0}'.format(quantile_regression_pd[cluster][cat]['slopes'][0.5]), linewidth=2, linestyle=stat_sig_pd[cluster][cat]['linestyle'][0.5], color='k')
	plt.plot(bins, quantile_regression_pd[cluster][cat]['predicts'][0.9], label=r'$\tau$ = 0.9, {0}'.format(quantile_regression_pd[cluster][cat]['slopes'][0.9]), linewidth=2, linestyle=stat_sig_pd[cluster][cat]['linestyle'][0.9], color='tab:blue')
	
	plt.xticks(np.arange(1915, 2021, 10))
	ax2.set_xticklabels(['1915','1925','1935','1945','1955','1965','1975','1985','1995','2005','2015'], fontsize=7)
	plt.yticks(y_ticks[cat])
	ax2.set_yticklabels(y_ticks[cat],fontsize=7)
	
	plt.grid()
	plt.gca().set_axisbelow(True)
		
	ax2.set_xlabel('Year', fontsize = 10)
	ax2.set_ylabel(y_labels[cat], fontsize = 10) #will change
	
	plt.legend(loc='upper left', fontsize=7, framealpha=1)

	plt.title(plot_titles[cat]['pd'], loc = "left",fontsize=10)
	plt.title('Pluvial-to-Drought', loc='right',fontsize=10)

	i+=2

plt.savefig(f'/home/bpuxley/Definition_and_Climatology/Plots/event_yearly_{cluster}.eps', bbox_inches = 'tight', pad_inches = 0.1)    


#########################################################################################
#Plot - Monthly Analysis (Figures 9 and 10)
#########################################################################################
#########################################################################################
#Add seasonal cycle of rainfall to this plot
#########################################################################################
clusters = ['cluster_1','cluster_2','cluster_3','cluster_4','cluster_5','cluster_6','cluster_7']
categories = ['freq', 'avg_size', 'avg_len'] # Define categories
bins=np.arange(1,13,1)

y_labels = {'freq': 'Number of Events', 
			'avg_size': 'Average Size of Event ($km^2$)', 
			'avg_len':'Average Length of Event ($\it{days}$)'}

y_ticks_conus = {'freq': np.arange(0, 160, 10), 
			'avg_size': np.arange(0, 550000, 50000), 
			'avg_len': np.arange(0, 11, 1)}
			
y_ticks = {'freq': np.arange(0, 45, 5), 
			'avg_size': np.arange(0, 550000, 50000), 
			'avg_len': np.arange(0, 11, 1)}

plot_titles = {'conus': 'a) CONUS',
				'cluster_1':'b) Cluster 1: Northwest', 
				'cluster_2':'c) Cluster 2: Northern Great Plains',
				'cluster_3':'d) Cluster 3: Southwest', 
				'cluster_4':'e) Cluster 4: Southern Great Plains',
				'cluster_5':'f) Cluster 5: Great Lakes', 
				'cluster_6':'g) Cluster 6: Southeast',
				'cluster_7':'h) Cluster 7: Northeast'
				}

for cat in categories:
	print(f"\nPlotting category: {cat}")
	i=1
	fig = plt.figure(figsize = (10,12), dpi = 300, tight_layout =True)
	
	cluster='conus'
	ax1 = fig.add_subplot(4,2,i)
	plt.bar(bins, df_dp_monthly_histograms[cluster][cat], color ='mediumpurple', alpha=1,  edgecolor ='k', width=0.4, linewidth=1, label = 'drought-to-pluvial')
	plt.bar(bins+0.4, df_pd_monthly_histograms[cluster][cat], color ='lightcoral', alpha=1,  edgecolor ='k', width=0.4, linewidth=1, label = 'pluvial-to-drought')
	
	plt.xticks(np.arange(1.2, 13.2, 1))
	ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov', 'Dec'], fontsize=7)
	plt.yticks(y_ticks_conus[cat])
	ax1.set_yticklabels(y_ticks_conus[cat],fontsize=7)
		
	ax2 = ax1.twinx()
	ax2.plot(bins+0.2, monthly_rainfall[cluster], color = 'black', lw ='2', label = 'Annual Rainfall')
	plt.yticks(np.arange(0,140,20))
	ax2.set_yticklabels(np.arange(0,140,20),fontsize=7)
	
	plt.grid()
	plt.gca().set_axisbelow(True)
	
	ax1.set_xlabel('Month', fontsize = 10)
	ax1.set_ylabel(y_labels[cat], fontsize = 10) #will change
	ax2.set_ylabel('Rainfall (mm)', fontsize = 10)
		
	# Combine legends
	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=2, fontsize=7, framealpha=1)

	plt.title(plot_titles[cluster], loc ='left')

	i+=1
	for cluster in clusters:
		print(f"\nPlotting cluster: {cluster}")
		ax1 = fig.add_subplot(4,2,i)
		plt.bar(bins, df_dp_monthly_histograms[cluster][cat], color ='mediumpurple', alpha=1,  edgecolor ='k', width=0.4, linewidth=1, label = 'drought-to-pluvial')
		plt.bar(bins+0.4, df_pd_monthly_histograms[cluster][cat], color ='lightcoral', alpha=1,  edgecolor ='k', width=0.4, linewidth=1, label = 'pluvial-to-drought')
	
		plt.xticks(np.arange(1.2, 13.2, 1))
		ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov', 'Dec'], fontsize=7)
		plt.yticks(y_ticks[cat])
		ax1.set_yticklabels(y_ticks[cat],fontsize=7)
		
		ax2 = ax1.twinx()
		ax2.plot(bins+0.2, monthly_rainfall[cluster], color = 'black', lw ='2', label = 'Annual Rainfall')
		plt.yticks(np.arange(0,140,20))
		ax2.set_yticklabels(np.arange(0,140,20),fontsize=7)
	
		plt.grid()
		plt.gca().set_axisbelow(True)
	
		ax1.set_xlabel('Month', fontsize = 10)
		ax1.set_ylabel(y_labels[cat], fontsize = 10) #will change
		ax2.set_ylabel('Rainfall (mm)', fontsize = 10)
		
		# Combine legends
		lines1, labels1 = ax1.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=2, fontsize=7, framealpha=1)

		plt.title(plot_titles[cluster], loc ='left')

		i+=1
	
	plt.savefig(f'/home/bpuxley/Definition_and_Climatology/Plots/event_monthly_{cat}.eps', bbox_inches = 'tight', pad_inches = 0.1)    


#########################################################################################
#Plot - Cluster Trends Spatially (Figure 8 and Supplementary Figure 5)
#########################################################################################
cluster_no = 7
q = 0.9 #or 0.5
categories = ['freq', 'avg_size', 'tot_area'] # Define categories

cmap = cm.RdBu_r
poly_colors = ['red', 'green', 'purple', 'saddlebrown', 'blue', 'black', 'orange', 'gold', 'pink', 'deeppink', 
			'deepskyblue', 'springgreen', 'olive', 'tan', 'grey', 'darkred', 'cyan', 'mediumpurple','tomato','chocolate',
			'yellow','lawngreen','lavender','plum','fuchsia','palevioletred','rosybrown','darkcyan','aquamarine','navy']

colorbar_limits = {'freq': {0.5: math.ceil(colorbar_limit_func(cluster_polys, 'freq', 0.5)*100)/100, 0.9: math.ceil(colorbar_limit_func(cluster_polys, 'freq', 0.9)*100)/100},
					'avg_size': {0.5: math.ceil(colorbar_limit_func(cluster_polys, 'avg_size', 0.5)/1000)*1000, 0.9: math.ceil(colorbar_limit_func(cluster_polys, 'avg_size', 0.9)/1000)*1000}, 
					'tot_area': {0.5: math.ceil(colorbar_limit_func(cluster_polys, 'tot_area', 0.5)/1000)*1000, 0.9: math.floor(colorbar_limit_func(cluster_polys, 'tot_area', 0.9)/1000)*1000},
					'avg_len': {0.5:  math.ceil(colorbar_limit_func(cluster_polys, 'avg_len', 0.5)*100)/100, 0.9: math.ceil(colorbar_limit_func(cluster_polys, 'avg_len', 0.9)*100)/100}
				}

colorbar_axis_title = {'freq': r'Trend per year ($\it{number}$)',
				'avg_size': 'Trend per year ($km^2$)',
				'tot_area': 'Trend per year ($km^2$)', 
				'avg_len': r'Trend per year ($\it{days}$)'
				}
				
colorbar_axis_ticks = {'freq': {0.5: [-0.03,-0.02,-0.01,0.00,0.01,0.02,0.03], 0.9: [-0.04,-0.03,-0.02,-0.01,0.00,0.01,0.02,0.03,0.04] },
				'avg_size': {0.5: [-3000,-2000,-1000,0,1000,2000,3000], 0.9: [-3000,-2000,-1000,0,1000,2000,3000]},
				'tot_area': {0.5: [-6000,-4000,-2000,0,2000,4000,6000], 0.9: [-8000,-6000,-4000,-2000,0,2000,4000,6000,8000]} 
				}
			

plot_titles = {'freq': {'dp': 'a) Number of Events \nper Year', 'pd': 'b) Number of Events \nper Year'},
				'avg_size': {'dp':'c) Average Size of an \nEvent ($km^2$)', 'pd': 'd) Average Size of an \nEvent ($km^2$)'},
				'tot_area': {'dp': 'e) Total Area Impacted \nper Year ($km^2$)', 'pd':'f) Total Area Impacted \nper Year ($km^2$)'}, 
				'avg_len': {'dp': 'g) Average Number of Days \nAbove the Areal Threshold', 'pd': 'h) Average Number of Days \nAbove the Areal Threshold'}	
				}


fig = plt.figure(figsize = (10,12), dpi = 300, tight_layout =True)
plot_no=1
for cat in categories:
	print(cat)
	colorbar_limit = colorbar_limits[cat][q]
	norm = mcolors.Normalize(vmin=-colorbar_limit, vmax=colorbar_limit)
	
	#Drought-to-Pluvial
	ax1 = fig.add_subplot(4,2,plot_no, projection=ccrs.PlateCarree())
	ax1.add_feature(cfeature.COASTLINE)
	ax1.add_feature(cfeature.BORDERS, linewidth=1)
	ax1.add_feature(cfeature.STATES, edgecolor='black')
	ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

	for i in range(0, cluster_no):
		polycolor = poly_colors[i]
		polygons = cluster_polys['avg_poly'][i]
		fill_value = cluster_polys[str(cat)+'_dp_'+str(q)][i]
		fillcolor = cmap(norm(fill_value))
		stipple = cluster_polys[str(cat)+'_dp_'+str(q)+'_sig'][i]  
		hatch = '...' if stipple else None 
    
		x, y = polygons.exterior.xy
		coords = list(zip(x, y))
		patch = MplPolygon(coords, closed=True,
                           facecolor=fillcolor, edgecolor='black', hatch=hatch, linewidth=1,
                           transform=ccrs.PlateCarree())
		ax1.add_patch(patch)
	
		# Get centroid and add label
		centroid = polygons.centroid
		cx, cy = centroid.x, centroid.y
	
		ax1.text(cx, cy, str(i+1), transform=ccrs.PlateCarree(),
				ha='center', va='center', fontsize=10,
				fontweight='bold', color=polycolor,
				bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1', alpha=1))		

	# Colorbar
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	plt.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8, aspect=30, ticks=colorbar_axis_ticks[cat][q], label=colorbar_axis_title[cat])

	plt.title(plot_titles[cat]['dp'], loc = "left",fontsize=10)
	plt.title("Drought-to-Pluvial", loc = 'right',fontsize=10)


	#Pluvial-to-Drought
	ax2 = fig.add_subplot(4,2,plot_no+1, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

	ax2.add_feature(cfeature.COASTLINE)
	ax2.add_feature(cfeature.BORDERS, linewidth=1)
	ax2.add_feature(cfeature.STATES, edgecolor='black')

	ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())
		
	for i in range(0, cluster_no):
		polycolor = poly_colors[i]
		polygons = cluster_polys['avg_poly'][i]
		fill_value = cluster_polys[str(cat)+'_pd_'+str(q)][i]
		fillcolor = cmap(norm(fill_value))
		stipple = cluster_polys[str(cat)+'_pd_'+str(q)+'_sig'][i]  
		hatch = '...' if stipple else None 

		x, y = polygons.exterior.xy
		coords = list(zip(x, y))
		patch = MplPolygon(coords, closed=True,
                           facecolor=fillcolor, edgecolor='black', hatch=hatch, linewidth=1,
                           transform=ccrs.PlateCarree())
		ax2.add_patch(patch)
	
		# Get centroid and add label
		centroid = polygons.centroid
		cx, cy = centroid.x, centroid.y
	
		ax2.text(cx, cy, str(i+1), transform=ccrs.PlateCarree(),
			ha='center', va='center', fontsize=10,
			fontweight='bold', color=polycolor,
			bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1', alpha=1))		

	# Colorbar
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	plt.colorbar(sm, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8, aspect=30,  ticks=colorbar_axis_ticks[cat][q], label=colorbar_axis_title[cat])

	plt.title(plot_titles[cat]['pd'], loc = "left",fontsize=10)
	plt.title("Pluvial-to-Drought", loc = 'right',fontsize=10)

	plot_no+=2
			
plt.savefig(f'/home/bpuxley/Definition_and_Climatology/Plots/cluster_regional_trends_{q}_new.eps', bbox_inches = 'tight', pad_inches = 0.1)    
