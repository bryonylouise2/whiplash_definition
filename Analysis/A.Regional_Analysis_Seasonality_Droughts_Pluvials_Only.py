#########################################################################################
## A script to calculate the seasonality of drought and pluvial only events throughout the 
## timeframe for individual clusters
## Bryony Louise Puxley
## Last Edited: Wednesday, May 28th
## Input: Independent event files of Drought only and Pluvial only events. 
## Output: PNG Files of 1) the temporal trends of drought and pluvial precipitation whiplash 
## events across CONUS, 2) Two PNGs of the temporal trends in the  50th percentile 
## (Supplementary Figure 6) and 90th percentiles of drought and pluvial only events for each 
## of the 7 clusters, and 3) the seasonal cycle of the frequency of drought and pluvial only
## events.
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
	date_column = df.columns[2]
	years = np.arange(1915,2021,1)
	months = np.arange(1,13,1)
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
	largest_area['year'] = pd.to_datetime(largest_area[date_column]).dt.year
	last_day['year'] = pd.to_datetime(last_day[date_column]).dt.year
	
	#Add a month only column to largest area dataframes
	largest_area['month'] = pd.to_datetime(largest_area[date_column]).dt.month
	last_day['month'] = pd.to_datetime(last_day[date_column]).dt.month
	
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
	
	# Reindex to include all years, filling missing ones with 0
	avg_monthly_size = avg_monthly_size.reindex(months, fill_value=0)

	return avg_size, total_area, avg_length, avg_monthly_size, avg_monthly_length

def colorbar_limit_func(df, cat, q):
	fill_values = df[[str(cat)+'_droughts_'+str(q), str(cat)+'_pluvials_'+str(q)]]
	min_value = fill_values.min().min()
	max_value = fill_values.max().max()
	colorbar_limit = abs(max([min_value, max_value], key=abs))
	
	return colorbar_limit
	
def start_dates(df):
	subset_begin_ind = np.where((df.Day_No == 0))[0]
	dates = df[df.columns[2]].iloc[subset_begin_ind].reset_index(drop=True)
	
	return dates

years = np.arange(1915,2021,1)


#########################################################################################
# Import Events - load previously made event files
#########################################################################################
events_droughts = pd.read_csv('/data2/bpuxley/droughts_and_pluvials/Events/independent_events_droughts.csv')
events_pluvials = pd.read_csv('/data2/bpuxley/droughts_and_pluvials/Events/independent_events_pluvials.csv')

df_droughts = events_droughts.copy()
df_pluvials = events_pluvials.copy()

cluster_droughts_databases = {'conus': df_droughts, 
						'cluster_1': df_droughts.iloc[np.where((df_droughts.cluster_no == 1))].reset_index(drop=True), 
						'cluster_2': df_droughts.iloc[np.where((df_droughts.cluster_no == 2))].reset_index(drop=True),
						'cluster_3': df_droughts.iloc[np.where((df_droughts.cluster_no == 3))].reset_index(drop=True), 
						'cluster_4': df_droughts.iloc[np.where((df_droughts.cluster_no == 4))].reset_index(drop=True), 
						'cluster_5': df_droughts.iloc[np.where((df_droughts.cluster_no == 5))].reset_index(drop=True), 
						'cluster_6': df_droughts.iloc[np.where((df_droughts.cluster_no == 6))].reset_index(drop=True), 
						'cluster_7': df_droughts.iloc[np.where((df_droughts.cluster_no == 7))].reset_index(drop=True) 
						}
						
cluster_pluvials_databases = {'conus': df_pluvials, 
						'cluster_1': df_pluvials.iloc[np.where((df_pluvials.cluster_no == 1))].reset_index(drop=True), 
						'cluster_2': df_pluvials.iloc[np.where((df_pluvials.cluster_no == 2))].reset_index(drop=True),
						'cluster_3': df_pluvials.iloc[np.where((df_pluvials.cluster_no == 3))].reset_index(drop=True), 
						'cluster_4': df_pluvials.iloc[np.where((df_pluvials.cluster_no == 4))].reset_index(drop=True), 
						'cluster_5': df_pluvials.iloc[np.where((df_pluvials.cluster_no == 5))].reset_index(drop=True), 
						'cluster_6': df_pluvials.iloc[np.where((df_pluvials.cluster_no == 6))].reset_index(drop=True), 
						'cluster_7': df_pluvials.iloc[np.where((df_pluvials.cluster_no == 7))].reset_index(drop=True) 
						}						

no_of_events_droughts = {'conus': len(cluster_droughts_databases['conus'].iloc[np.where((cluster_droughts_databases['conus'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_1': len(cluster_droughts_databases['cluster_1'].iloc[np.where((cluster_droughts_databases['cluster_1'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_2': len(cluster_droughts_databases['cluster_2'].iloc[np.where((cluster_droughts_databases['cluster_2'].Day_No == 0))].reset_index(drop=True)),
					'cluster_3': len(cluster_droughts_databases['cluster_3'].iloc[np.where((cluster_droughts_databases['cluster_3'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_4': len(cluster_droughts_databases['cluster_4'].iloc[np.where((cluster_droughts_databases['cluster_4'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_5': len(cluster_droughts_databases['cluster_5'].iloc[np.where((cluster_droughts_databases['cluster_5'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_6': len(cluster_droughts_databases['cluster_6'].iloc[np.where((cluster_droughts_databases['cluster_6'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_7': len(cluster_droughts_databases['cluster_7'].iloc[np.where((cluster_droughts_databases['cluster_7'].Day_No == 0))].reset_index(drop=True))
					 }

no_of_events_pluvials = {'conus': len(cluster_pluvials_databases['conus'].iloc[np.where((cluster_pluvials_databases['conus'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_1': len(cluster_pluvials_databases['cluster_1'].iloc[np.where((cluster_pluvials_databases['cluster_1'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_2': len(cluster_pluvials_databases['cluster_2'].iloc[np.where((cluster_pluvials_databases['cluster_2'].Day_No == 0))].reset_index(drop=True)),
					'cluster_3': len(cluster_pluvials_databases['cluster_3'].iloc[np.where((cluster_pluvials_databases['cluster_3'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_4': len(cluster_pluvials_databases['cluster_4'].iloc[np.where((cluster_pluvials_databases['cluster_4'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_5': len(cluster_pluvials_databases['cluster_5'].iloc[np.where((cluster_pluvials_databases['cluster_5'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_6': len(cluster_pluvials_databases['cluster_6'].iloc[np.where((cluster_pluvials_databases['cluster_6'].Day_No == 0))].reset_index(drop=True)), 
					'cluster_7': len(cluster_pluvials_databases['cluster_7'].iloc[np.where((cluster_pluvials_databases['cluster_7'].Day_No == 0))].reset_index(drop=True))
					 }


#########################################################################################
#Calculate the frequency of the events throughout the time frame
#########################################################################################
clusters = ['conus','cluster_1','cluster_2','cluster_3','cluster_4','cluster_5','cluster_6','cluster_7']

#create empty dictionaries to store variables.
#yearly - frequency, avg_size, total_area, avg_len
df_droughts_yearly_histograms = {}
df_pluvials_yearly_histograms = {}

#monthly - frequency, avg_size, avg_len
df_droughts_monthly_histograms = {}
df_pluvials_monthly_histograms = {}

#quantile regression
quantile_regression_droughts = {}
quantile_regression_pluvials = {}

#statistical significance
stat_sig_droughts = {}
stat_sig_pluvials = {}

#correlation with enso
enso_corrs_droughts = {}
enso_corrs_pluvials = {}

#means of each half
means_droughts = {}
means_pluvials = {}

#permutation test results
permutation_results_droughts = {}
permutation_results_pluvials = {}

#quantile regression
quantile_regression_first_half_droughts = {}
quantile_regression_first_half_pluvials = {}

quantile_regression_second_half_droughts = {}
quantile_regression_second_half_pluvials = {}

#statistical significance
stat_sig_first_half_droughts = {}
stat_sig_first_half_pluvials = {}

stat_sig_second_half_droughts = {}
stat_sig_second_half_pluvials = {}

for cluster in clusters:
	print(f"\nProcessing: {cluster}")
	
	df_droughts = cluster_droughts_databases[cluster]
	df_pluvials = cluster_pluvials_databases[cluster]
	
	#########################################################################################
	# Create a temp event number for analysis
	#########################################################################################
	day0_only_droughts = df_droughts.iloc[np.where((df_droughts.Day_No == 0))].reset_index(drop=True)
	day0_only_pluvials = df_pluvials.iloc[np.where((df_pluvials.Day_No == 0))].reset_index(drop=True)
	
	mapping_droughts = dict(zip(day0_only_droughts['Event_No'], day0_only_droughts.index+1))
	mapping_pluvials = dict(zip(day0_only_pluvials['Event_No'], day0_only_pluvials.index+1))

	df_droughts['event_no_temp'] = [mapping_droughts[val] for val in df_droughts['Event_No']]
	df_pluvials['event_no_temp'] = [mapping_pluvials[val] for val in df_pluvials['Event_No']]
	
	#########################################################################################
	#Calculate the frequency of the events throughout the time frame
	#########################################################################################
	#Extract the year from the start dates of the events and create a histogram
	start_dates_droughts = start_dates(df_droughts)
	start_dates_pluvials = start_dates(df_pluvials)

	years_droughts = pd.to_datetime(start_dates_droughts).dt.year
	years_pluvials = pd.to_datetime(start_dates_pluvials).dt.year

	hist_yearly_freq_droughts = np.histogram(years_droughts, bins=np.arange(1915,2022,1))[0]
	hist_yearly_freq_pluvials = np.histogram(years_pluvials, bins=np.arange(1915,2022,1))[0]

	#########################################################################################
	#Calculate the frequency of the events throughout the year
	#########################################################################################
	months_droughts =  pd.to_datetime(start_dates_droughts).dt.month
	months_pluvials =  pd.to_datetime(start_dates_pluvials).dt.month

	hist_monthly_freq_droughts = np.histogram(months_droughts, bins=np.arange(1,14,1))[0]
	hist_monthly_freq_pluvials = np.histogram(months_pluvials, bins=np.arange(1,14,1))[0]
	
	#########################################################################################
	#Calculate how the areal size of events and length of events are changing throughout time
	#########################################################################################
	hist_yearly_avg_size_droughts, hist_yearly_tot_area_droughts, hist_yearly_avg_len_droughts, hist_monthly_avg_size_droughts, hist_monthly_avg_len_droughts = area_and_length(df_droughts)
	hist_yearly_avg_size_pluvials, hist_yearly_tot_area_pluvials, hist_yearly_avg_len_pluvials, hist_monthly_avg_size_pluvials, hist_monthly_avg_len_pluvials = area_and_length(df_pluvials)

	#########################################################################################
	#Assign to dictionaries
	#########################################################################################
	df_droughts_yearly_histograms[cluster] = pd.DataFrame({'freq': hist_yearly_freq_droughts, 'avg_size':hist_yearly_avg_size_droughts, 'tot_area':hist_yearly_tot_area_droughts})
	df_pluvials_yearly_histograms[cluster] = pd.DataFrame({'freq': hist_yearly_freq_pluvials, 'avg_size':hist_yearly_avg_size_pluvials, 'tot_area':hist_yearly_tot_area_pluvials})
 
	df_droughts_monthly_histograms[cluster] = pd.DataFrame({'freq': hist_monthly_freq_droughts, 'avg_size':hist_monthly_avg_size_droughts})
	df_pluvials_monthly_histograms[cluster] = pd.DataFrame({'freq': hist_monthly_freq_pluvials, 'avg_size':hist_monthly_avg_size_pluvials})

	#########################################################################################
	#Calculate the quantile regression for all plots (yearly)
	#########################################################################################
	categories = ['freq', 'avg_size', 'tot_area'] # Define categories
	quantiles = [0.1, 0.5, 0.9] # Define quantiles

	quantile_regression_droughts[cluster] = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(df_droughts_yearly_histograms[cluster].index, df_droughts_yearly_histograms[cluster][cat], quantiles)] }
	quantile_regression_pluvials[cluster] = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(df_pluvials_yearly_histograms[cluster].index, df_pluvials_yearly_histograms[cluster][cat], quantiles)] }

	#########################################################################################
	#Calculate the statistical significance using using bias-corrected and accelerated (BCa) 
	#bootstrapping (Efron 1987) with 10 000 replications at p < 0.05 and a null hypothesis of 
	#no trend. 
	#########################################################################################
	niter = 10000 #Number of Iterations
	confidence_level = 0.95 #confidence level
	alpha = round(1 - confidence_level, 2)

	stat_sig_droughts[cluster] = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4, 'sig':val5, 'linestyle':val6} for cat in categories for val1, val2, val3, val4, val5, val6 in [functions.bca_bootstrap(df_droughts_yearly_histograms[cluster].index, df_droughts_yearly_histograms[cluster][cat], quantiles, niter, alpha)] }
	stat_sig_pluvials[cluster] = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4, 'sig':val5, 'linestyle':val6} for cat in categories for val1, val2, val3, val4, val5, val6 in [functions.bca_bootstrap(df_pluvials_yearly_histograms[cluster].index, df_pluvials_yearly_histograms[cluster][cat], quantiles, niter, alpha)] }

	#########################################################################################
	#Correlation between ENSO years and events
	#########################################################################################	
	enso_corrs_droughts[cluster] = {cat: {'statistics':round(val1,3), 'pvalue':round(val2,3)} for cat in categories for val1, val2 in [scs.pearsonr(enso_years.avg_oni, df_droughts_yearly_histograms[cluster][cat])]}
	enso_corrs_pluvials[cluster] = {cat: {'statistics':round(val1,3), 'pvalue':round(val2,3)} for cat in categories for val1, val2 in [scs.pearsonr(enso_years.avg_oni, df_pluvials_yearly_histograms[cluster][cat])]}

	#########################################################################################
	#Split Database into 2: (1915-1967)/(1968-2020)
	#########################################################################################	
	first_half_droughts = df_droughts_yearly_histograms[cluster][:53]
	second_half_droughts = df_droughts_yearly_histograms[cluster][53:]

	first_half_pluvials = df_pluvials_yearly_histograms[cluster][:53]
	second_half_pluvials = df_pluvials_yearly_histograms[cluster][53:]

	#Calculate the means of each period
	means_droughts[cluster] = {cat: {'first_half': round(np.nanmean(first_half_droughts[cat]),3), 'second_half': round(np.nanmean(second_half_droughts[cat]),3)} for cat in categories}
	means_pluvials[cluster] = {cat: {'first_half': round(np.nanmean(first_half_pluvials[cat]),3), 'second_half': round(np.nanmean(second_half_pluvials[cat]),3)} for cat in categories}

	#Permutation Test
	permutation_results_droughts[cluster] = {cat: {'difference':val1, 'pvalue':val2} for cat in categories for val1, val2 in [functions.permutation_test(np.array(first_half_droughts[cat]), np.array(second_half_droughts[cat]), 10000)]}
	permutation_results_pluvials[cluster] = {cat: {'difference':val1, 'pvalue':val2} for cat in categories for val1, val2 in [functions.permutation_test(np.array(first_half_pluvials[cat]), np.array(second_half_pluvials[cat]), 10000)]}

	#Quantile Regression on each separate half
	quantile_regression_first_half_droughts[cluster] = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(first_half_droughts.index, first_half_droughts[cat], quantiles)] }
	quantile_regression_second_half_droughts[cluster] = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(second_half_droughts.index, second_half_droughts[cat], quantiles)] }

	quantile_regression_first_half_pluvials[cluster] = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(first_half_pluvials.index, first_half_pluvials[cat], quantiles)] }
	quantile_regression_second_half_pluvials[cluster] = {cat: {'predicts':val1, 'slopes':val2} for cat in categories for val1, val2 in [functions.QuantileRegression(second_half_pluvials.index, second_half_pluvials[cat], quantiles)] }

	#Significance Testing on each separate half
	stat_sig_first_half_droughts[cluster] = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4} for cat in categories for val1, val2, val3, val4 in [functions.bca_bootstrap(first_half_droughts.index, first_half_droughts[cat], quantiles, niter, alpha)] }
	stat_sig_second_half_droughts[cluster] = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4} for cat in categories for val1, val2, val3, val4 in [functions.bca_bootstrap(second_half_droughts.index, second_half_droughts[cat], quantiles, niter, alpha)] }

	stat_sig_first_half_pluvials[cluster] = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4} for cat in categories for val1, val2, val3, val4 in [functions.bca_bootstrap(first_half_pluvials.index, first_half_pluvials[cat], quantiles, niter, alpha)] }
	stat_sig_second_half_pluvials[cluster] = {cat: {'slopes':val1, 'lower_bounds':val2, 'upper_bounds':val3, 'pvals':val4} for cat in categories for val1, val2, val3, val4 in [functions.bca_bootstrap(second_half_pluvials.index, second_half_pluvials[cat], quantiles, niter, alpha)] }

#########################################################################################
#Read in Cluster Data
#########################################################################################
clusters = ['cluster_1','cluster_2','cluster_3','cluster_4','cluster_5','cluster_6','cluster_7']
categories = ['freq', 'avg_size', 'tot_area'] # Define categories
quantiles= [0.5,0.9]

cluster_polys = pd.read_csv('/data2/bpuxley/Events/cluster_polygons_final.csv')

cluster_polys['avg_poly'] = cluster_polys['avg_poly'].apply(shapely.wkt.loads)
cluster_polys = cluster_polys.drop('7', axis=1)

cluster_polys[['freq_droughts_0.5', 
				'freq_droughts_0.5_sig',
				'freq_droughts_0.9', 
				'freq_droughts_0.9_sig',
				'freq_pluvials_0.5', 
				'freq_pluvials_0.5_sig',
				'freq_pluvials_0.9', 
				'freq_pluvials_0.9_sig',
				
				'avg_size_droughts_0.5', 
				'avg_size_droughts_0.5_sig',
				'avg_size_droughts_0.9', 
				'avg_size_droughts_0.9_sig',
				'avg_size_pluvials_0.5', 
				'avg_size_pluvials_0.5_sig',
				'avg_size_pluvials_0.9', 
				'avg_size_pluvials_0.9_sig',
				
				'tot_area_droughts_0.5', 
				'tot_area_droughts_0.5_sig',
				'tot_area_droughts_0.9', 
				'tot_area_droughts_0.9_sig',
				'tot_area_pluvials_0.5', 
				'tot_area_pluvials_0.5_sig',
				'tot_area_pluvials_0.9', 
				'tot_area_pluvials_0.9_sig',
				
				'avg_len_droughts_0.5', 
				'avg_len_droughts_0.5_sig',
				'avg_len_droughts_0.9', 
				'avg_len_droughts_0.9_sig',
				'avg_len_pluvials_0.5', 
				'avg_len_pluvials_0.5_sig',
				'avg_len_pluvials_0.9', 
				'avg_len_pluvials_0.9_sig',
				]] = ''

#add columns to cluster_poly to represent trends and significance
for cat in categories:
	print(cat)
	for q in quantiles:
		print(q)
		i=0
		for cluster in clusters:
			#Slope
			cluster_polys.loc[i, str(cat)+'_droughts_'+str(q)] = quantile_regression_droughts[cluster][cat]['slopes'][q]
			cluster_polys.loc[i, str(cat)+'_pluvials_'+str(q)] = quantile_regression_pluvials[cluster][cat]['slopes'][q]
			
			#Significance
			cluster_polys.loc[i, str(cat)+'_droughts_'+str(q)+'_sig'] = True if stat_sig_droughts[cluster][cat]['pvals'][q] <= 0.05 else False
			cluster_polys.loc[i, str(cat)+'_pluvials_'+str(q)+'_sig'] = True if stat_sig_pluvials[cluster][cat]['pvals'][q] <= 0.05 else False
			
			i+=1

cluster_polys['freq_droughts_0.5_sig'] = [False, False, True, False, False, False, False]
cluster_polys['freq_droughts_0.9_sig'] = [False, False, True, False, False, False, True]

cluster_polys['freq_pluvials_0.5_sig'] = [False, False, False, False, True, False, False]
cluster_polys['freq_pluvials_0.9_sig'] = [True, False, False, True, False, False, True]

cluster_polys['avg_size_droughts_0.5_sig'] = [False, False, False, False, False, False, False]
cluster_polys['avg_size_droughts_0.9_sig'] = [False, False, False, False, False, False, False]

cluster_polys['avg_size_pluvials_0.5_sig'] = [False, False, False, False, False, False, False]
cluster_polys['avg_size_pluvials_0.9_sig'] = [False, False, False, False, False, False, False]

cluster_polys['tot_area_droughts_0.5_sig'] = [False, False, True, False, False, False, False]
cluster_polys['tot_area_droughts_0.9_sig'] = [False, False, False, False, False, False, False]

cluster_polys['tot_area_pluvials_0.5_sig'] = [False, False, False, False, True, False, False]
cluster_polys['tot_area_pluvials_0.9_sig'] = [False, False, False, False, False, False, False]


cluster_polys.to_csv('/data2/bpuxley/droughts_and_pluvials/Events/temporal_trends.csv')

#########################################################################################
#########################################################################################
#Plots
#########################################################################################
#########################################################################################
#########################################################################################
#Plot - Yearly Analysis (one plot)
#########################################################################################
cluster = 'cluster_7'
categories = ['freq', 'avg_size', 'tot_area'] # Define categories
bins=np.arange(1915,2021,1)

y_labels = {'freq': 'Number of Events', 
			'avg_size': 'Average Size of Event ($km^2$)', 
			'tot_area':'Total Area Impacted ($km^2$)'}

y_ticks = {'freq': np.arange(0, 14, 1), 
			'avg_size': np.arange(0, 1800000, 100000), 
			'tot_area':np.arange(0, 6000000, 500000)}

plot_titles = {'freq': {'dp': 'a) Number of Events \nper Year', 'pd': 'b) Number of Events \nper Year'},
				'avg_size': {'dp':'c) Average Size of an \nEvent ($km^2$)', 'pd': 'd) Average Size of an \nEvent ($km^2$)'},
				'tot_area': {'dp': 'e) Total Area Impacted \nper Year ($km^2$)', 'pd':'f) Total Area Impacted \nper Year ($km^2$)'}}

fig = plt.figure(figsize = (10,12), dpi = 300, tight_layout =True)
i=1
for cat in categories:
	print(f"\nPlotting category: {cat}")
	
	# Drought-to-Pluvial
	print(i)
	ax1 = fig.add_subplot(3,2,i)
	
	plt.bar(bins, df_droughts_yearly_histograms[cluster][cat], color = 'mediumpurple', alpha=1,  edgecolor ='k', width=1, linewidth=1)
	plt.plot(bins, quantile_regression_droughts[cluster][cat]['predicts'][0.1], label=r'$\tau$ = 0.1, {0}, [{1}, {2}]'.format(quantile_regression_droughts[cluster][cat]['slopes'][0.1], stat_sig_droughts[cluster][cat]['lower_bounds'][0.1], stat_sig_droughts[cluster][cat]['upper_bounds'][0.1]), linewidth=2, color='darkgreen')
	plt.plot(bins, quantile_regression_droughts[cluster][cat]['predicts'][0.5], label=r'$\tau$ = 0.5, {0}, [{1}, {2}]'.format(quantile_regression_droughts[cluster][cat]['slopes'][0.5], stat_sig_droughts[cluster][cat]['lower_bounds'][0.5], stat_sig_droughts[cluster][cat]['upper_bounds'][0.5]), linewidth=2, color='k')
	plt.plot(bins, quantile_regression_droughts[cluster][cat]['predicts'][0.9], label=r'$\tau$ = 0.9, {0}, [{1}, {2}]'.format(quantile_regression_droughts[cluster][cat]['slopes'][0.9], stat_sig_droughts[cluster][cat]['lower_bounds'][0.9], stat_sig_droughts[cluster][cat]['upper_bounds'][0.9]), linewidth=2, color='tab:blue')
	
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
	plt.title("Drought Events", loc = 'right',fontsize=10)


	# Pluvial to Drought
	print(i+1)
	ax2 = fig.add_subplot(3,2,i+1)
	
	plt.bar(bins, df_pluvials_yearly_histograms[cluster][cat], color = 'lightcoral', alpha=1,  edgecolor ='k', width=1, linewidth=1)
	plt.plot(bins, quantile_regression_pluvials[cluster][cat]['predicts'][0.1], label=r'$\tau$ = 0.1, {0}, [{1}, {2}]'.format(quantile_regression_pluvials[cluster][cat]['slopes'][0.1], stat_sig_pluvials[cluster][cat]['lower_bounds'][0.1], stat_sig_pluvials[cluster][cat]['upper_bounds'][0.1]), linewidth=2, color='darkgreen')
	plt.plot(bins, quantile_regression_pluvials[cluster][cat]['predicts'][0.5], label=r'$\tau$ = 0.5, {0}, [{1}, {2}]'.format(quantile_regression_pluvials[cluster][cat]['slopes'][0.5], stat_sig_pluvials[cluster][cat]['lower_bounds'][0.5], stat_sig_pluvials[cluster][cat]['upper_bounds'][0.5]), linewidth=2, color='k')
	plt.plot(bins, quantile_regression_pluvials[cluster][cat]['predicts'][0.9], label=r'$\tau$ = 0.9, {0}, [{1}, {2}]'.format(quantile_regression_pluvials[cluster][cat]['slopes'][0.9], stat_sig_pluvials[cluster][cat]['lower_bounds'][0.9], stat_sig_pluvials[cluster][cat]['upper_bounds'][0.9]), linewidth=2, color='tab:blue')
	
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
	plt.title('Pluvial Events', loc='right',fontsize=10)

	i+=2

plt.savefig(f'/home/bpuxley/Definition_and_Climatology/Plots/event_yearly_{cluster}_droughts_pluvials.png', bbox_inches = 'tight', pad_inches = 0.1)    


#########################################################################################
#Plot - Monthly Analysis (frequency)
#########################################################################################
#########################################################################################
#Add seasonal cycle of rainfall to this plot
#########################################################################################
clusters = ['conus','cluster_1','cluster_2','cluster_3','cluster_4','cluster_5','cluster_6','cluster_7']
categories = ['freq', 'avg_size'] # Define categories
bins=np.arange(1,13,1)

y_labels = {'freq': 'Number of Events', 
			'avg_size': 'Average Size of Event ($km^2$)'}

y_ticks = {'freq': np.arange(0, 160, 10), 
			'avg_size': np.arange(0, 550000, 50000)}

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
	for cluster in clusters:
		print(f"\nPlotting cluster: {cluster}")
		ax1 = fig.add_subplot(4,2,i)
		plt.bar(bins, df_droughts_monthly_histograms[cluster][cat], color ='mediumpurple', alpha=1,  edgecolor ='k', width=0.4, linewidth=1, label = 'droughts')
		plt.bar(bins+0.4, df_pluvials_monthly_histograms[cluster][cat], color ='lightcoral', alpha=1,  edgecolor ='k', width=0.4, linewidth=1, label = 'pluvials')
	
		plt.xticks(np.arange(1.2, 13.2, 1))
		ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov', 'Dec'], fontsize=7)
		plt.yticks(y_ticks[cat])
		ax1.set_yticklabels(y_ticks[cat],fontsize=7)
	
		plt.grid()
		plt.gca().set_axisbelow(True)
	
		ax1.set_xlabel('Month', fontsize = 10)
		ax1.set_ylabel(y_labels[cat], fontsize = 10) #will change
		
		plt.legend(loc='upper left', fontsize=7, framealpha=1)
	
		plt.title(plot_titles[cluster], loc ='left')

		i+=1
	
	plt.savefig(f'/home/bpuxley/Definition_and_Climatology/Plots/event_monthly_{cat}_droughts_pluvials.png', bbox_inches = 'tight', pad_inches = 0.1)    


#########################################################################################
#Plot - Cluster Trends Spatially
#########################################################################################
cluster_no = 7
q = 0.5 #or 0.9
categories = ['freq', 'avg_size', 'tot_area'] # Define categories

cmap = cm.RdBu_r
poly_colors = ['red', 'green', 'purple', 'saddlebrown', 'blue', 'black', 'orange', 'gold', 'pink', 'deeppink', 
			'deepskyblue', 'springgreen', 'olive', 'tan', 'grey', 'darkred', 'cyan', 'mediumpurple','tomato','chocolate',
			'yellow','lawngreen','lavender','plum','fuchsia','palevioletred','rosybrown','darkcyan','aquamarine','navy']

colorbar_limits = {'freq': {0.5: math.ceil(colorbar_limit_func(cluster_polys, 'freq', 0.5)*100)/100, 0.9: math.ceil(colorbar_limit_func(cluster_polys, 'freq', 0.9)*100)/100},
					'avg_size': {0.5: math.ceil(colorbar_limit_func(cluster_polys, 'avg_size', 0.5)/1000)*1000, 0.9: math.ceil(colorbar_limit_func(cluster_polys, 'avg_size', 0.9)/1000)*1000}, 
					'tot_area': {0.5: math.ceil(colorbar_limit_func(cluster_polys, 'tot_area', 0.5)/1000)*1000, 0.9: math.floor(colorbar_limit_func(cluster_polys, 'tot_area', 0.9)/1000)*1000}
					}

colorbar_axis_title = {'freq': r'Trend per year ($\it{number}$)',
				'avg_size': 'Trend per year ($km^2$)',
				'tot_area': 'Trend per year ($km^2$)'
				}
				
colorbar_axis_ticks = {'freq': {0.5: [-0.03,-0.02,-0.01,0.00,0.01,0.02,0.03], 0.9: [-0.03,-0.02,-0.01,0.00,0.01,0.02,0.03] },
				'avg_size': {0.5: [-3000,-2000,-1000,0,1000,2000,3000], 0.9: [-3000,-2000,-1000,0,1000,2000,3000]},
				'tot_area': {0.5: [-9000,-4500,0,4500,9000], 0.9: [-17000,-8500,0,8500,17000]} 
				}

plot_titles = {'freq': {'droughts': 'a) Number of Events \nper Year', 'pluvials': 'b) Number of Events \nper Year'},
				'avg_size': {'droughts':'c) Average Size of an \nEvent ($km^2$)', 'pluvials': 'd) Average Size of an \nEvent ($km^2$)'},
				'tot_area': {'droughts': 'e) Total Area Impacted \nper Year ($km^2$)', 'pluvials':'f) Total Area Impacted \nper Year ($km^2$)'}	
				}


fig = plt.figure(figsize = (8,8), dpi = 300, tight_layout =True)
plot_no=1
for cat in categories:
	print(cat)
	colorbar_limit = colorbar_limits[cat][q]
	norm = mcolors.Normalize(vmin=-colorbar_limit, vmax=colorbar_limit)
	
	#Droughts
	ax1 = fig.add_subplot(3,2,plot_no, projection=ccrs.PlateCarree())
	ax1.add_feature(cfeature.COASTLINE)
	ax1.add_feature(cfeature.BORDERS, linewidth=1)
	ax1.add_feature(cfeature.STATES, edgecolor='black')
	ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())

	for i in range(0, cluster_no):
		polycolor = poly_colors[i]
		polygons = cluster_polys['avg_poly'][i]
		fill_value = cluster_polys[str(cat)+'_droughts_'+str(q)][i]
		fillcolor = cmap(norm(fill_value))
		stipple = cluster_polys[str(cat)+'_droughts_'+str(q)+'_sig'][i]  
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

	plt.title(plot_titles[cat]['droughts'], loc = "left",fontsize=10)
	plt.title("Drought Events", loc = 'right',fontsize=10)


	#Pluvials
	ax2 = fig.add_subplot(3,2,plot_no+1, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

	ax2.add_feature(cfeature.COASTLINE)
	ax2.add_feature(cfeature.BORDERS, linewidth=1)
	ax2.add_feature(cfeature.STATES, edgecolor='black')

	ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())
		
	for i in range(0, cluster_no):
		polycolor = poly_colors[i]
		polygons = cluster_polys['avg_poly'][i]
		fill_value = cluster_polys[str(cat)+'_pluvials_'+str(q)][i]
		fillcolor = cmap(norm(fill_value))
		stipple = cluster_polys[str(cat)+'_pluvials_'+str(q)+'_sig'][i]  
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

	plt.title(plot_titles[cat]['pluvials'], loc = "left",fontsize=10)
	plt.title("Pluvial Events", loc = 'right',fontsize=10)

	plot_no+=2
			
plt.savefig(f'/home/bpuxley/Definition_and_Climatology/Plots/cluster_regional_trends_{q}_droughts_pluvials.eps', bbox_inches = 'tight', pad_inches = 0.1)    

