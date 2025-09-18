#########################################################################################
## A script to examine and plot how long an event was in drought/pluvial conditions prior 
## to and after the event
## Bryony Louise Puxley
## Last Edited: Monday, August 11th, 2025 
## Input: Independent event files of Drought-to-Pluvial, Pluvial-to-Drought, Drought only,
## and Pluvial only events.
## Output: A scatter plot (PNG) of (a,c) Time in drought (SPI below zero) prior to vs time 
## in pluvial (SPI above zero) after our recorded drought-to-pluvial whiplash events, and 
## (b,d) time in pluvial (SPI above zero) prior to vs time in drought (SPI below zero) after
## our recorded pluvial-to-drought whiplash events (Figure 11) and the expanded full version
## (Supplementary Figure 7). Additionally, the values for the average number of days in 
## drought (SPI below zero) and pluvial (SPI above zero) conditions for each of our defined 
## clusters as shown in Table 2.
#########################################################################################
#Import Required Modules
#########################################################################################
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from itertools import cycle
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#########################################################################################
#Import Functions
#########################################################################################
import functions

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

#########################################################################################
#Import Events - load previously made event files
#########################################################################################
events_DP = pd.read_csv('/data2/bpuxley/Events/independent_events_DP.csv')
events_PD = pd.read_csv('/data2/bpuxley/Events/independent_events_PD.csv')
events_droughts = pd.read_csv('/data2/bpuxley/Events/independent_events_droughts.csv')
events_pluvials = pd.read_csv('/data2/bpuxley/Events/independent_events_pluvials.csv')

df_DP = events_DP.copy()
df_PD = events_PD.copy()
df_droughts = events_droughts.copy()
df_pluvials = events_pluvials.copy()

no_of_events_DP = np.nanmax(df_DP.Event_No)
no_of_events_PD = np.nanmax(df_PD.Event_No)
no_of_events_droughts = np.nanmax(df_droughts.Event_No)
no_of_events_pluvials = np.nanmax(df_pluvials.Event_No)

years = np.arange(1915,2021,1)


#########################################################################################
#Intensity
#########################################################################################
intensity_dp = intensity(df_DP)
intensity_pd = intensity(df_PD)

bins = [1, 2, 3, 4, 4.5]  # Adjust as needed
bin_labels = [f"{bins[i]} - {bins[i+1]}" for i in range(len(bins)-1)]

intensity_dp['Avg_SPI_Change_bins'] = pd.cut(intensity_dp['Avg_SPI_Change'], bins=bins, right=False)
intensity_pd['Avg_SPI_Change_bins'] = pd.cut(intensity_pd['Avg_SPI_Change'], bins=bins, right=False)

intensity_averages_DP = pd.DataFrame({'Intensity_bins': bin_labels, 'Time_Before': intensity_dp.groupby('Avg_SPI_Change_bins')['Time_Before'].mean().reset_index(drop=True), 'Time_After': intensity_dp.groupby('Avg_SPI_Change_bins')['Time_After'].mean().reset_index(drop=True)})
intensity_averages_PD = pd.DataFrame({'Intensity_bins': bin_labels, 'Time_Before': intensity_pd.groupby('Avg_SPI_Change_bins')['Time_Before'].mean().reset_index(drop=True), 'Time_After': intensity_pd.groupby('Avg_SPI_Change_bins')['Time_After'].mean().reset_index(drop=True)})


#########################################################################################
#Add a seasonal column
#########################################################################################
df_DP['month'] = pd.to_datetime(df_DP.Whiplash_Date).dt.month
df_PD['month']  = pd.to_datetime(df_PD.Whiplash_Date).dt.month

def categorize_season(value):
	if value == 12 or value == 1 or value == 2:
		return 1
	if value == 3 or value == 4 or value == 5:
		return 2
	if value == 6 or value == 7 or value == 8:
		return 3
	else:
		return 4
		
df_DP['season'] = df_DP['month'].apply(categorize_season)
df_PD['season'] = df_PD['month'].apply(categorize_season)

#########################################################################################
#Number of Events and Averages
#########################################################################################
#day_0_only
df_DP_new = df_DP[df_DP.Day_No ==0].reset_index(drop=True) 
df_PD_new = df_PD[df_PD.Day_No ==0].reset_index(drop=True)

df_droughts_new = df_droughts[df_droughts.Day_No ==0].reset_index(drop=True) 
df_pluvials_new = df_pluvials[df_pluvials.Day_No ==0].reset_index(drop=True)



#########################################################################################
#event length statistics
#########################################################################################
dp_statistics = {'mean_drought_prior': round(np.nanmean(df_DP_new.Time_in_Drought_Before+29),2),
					'max_drought_prior': round(np.nanmax(df_DP_new.Time_in_Drought_Before+29),2),
					'min_drought_prior': round(np.nanmin(df_DP_new.Time_in_Drought_Before+29),2),
					'mean_pluvial_after': round(np.nanmean(df_DP_new.Time_in_Pluvial_After+29),2),
					'max_pluvial_after': round(np.nanmax(df_DP_new.Time_in_Pluvial_After+29),2),
					'min_pluvial_after': round(np.nanmin(df_DP_new.Time_in_Pluvial_After+29),2)
					}
					
pd_statistics = {'mean_pluvial_prior': round(np.nanmean(df_PD_new.Time_in_Pluvial_Before+29),2),
					'max_pluvial_prior': round(np.nanmax(df_PD_new.Time_in_Pluvial_Before+29),2),
					'min_pluvial_prior': round(np.nanmin(df_PD_new.Time_in_Pluvial_Before+29),2),
					'mean_drought_after': round(np.nanmean(df_PD_new.Time_in_Drought_After+29),2),
					'max_drought_after': round(np.nanmax(df_PD_new.Time_in_Drought_After+29),2),
					'min_drought_after': round(np.nanmin(df_PD_new.Time_in_Drought_After+29),2)
					}

#########################################################################################
#Cluster
#########################################################################################
clusters = {'Cluster_1':1,'Cluster_2':2,'Cluster_3':3,'Cluster_4':4,'Cluster_5':5,'Cluster_6':6,'Cluster_7':7}

cluster_number_DP = {key: len(df_DP_new[df_DP_new.cluster_no == value].reset_index(drop=True)) for key, value in clusters.items()} 
cluster_number_PD = {key: len(df_PD_new[df_PD_new.cluster_no == value].reset_index(drop=True)) for key, value in clusters.items()} 
cluster_number_droughts = {key: len(df_droughts_new[df_droughts_new.cluster_no == value].reset_index(drop=True)) for key, value in clusters.items()} 
cluster_number_pluvials = {key: len(df_pluvials_new[df_pluvials_new.cluster_no == value].reset_index(drop=True)) for key, value in clusters.items()} 

cluster_averages_DP = {key: [round(np.nanmean(df_DP_new[df_DP_new.cluster_no == value].Time_in_Drought_Before),3),round(np.nanmean(df_DP_new[df_DP_new.cluster_no == value].Time_in_Pluvial_After),3)] for key, value in clusters.items()} 
cluster_averages_PD = {key: [round(np.nanmean(df_PD_new[df_PD_new.cluster_no == value].Time_in_Pluvial_Before),3),round(np.nanmean(df_PD_new[df_PD_new.cluster_no == value].Time_in_Drought_After),3)] for key, value in clusters.items()} 

cluster_averages_DP = {key: [round(np.nanmean(df_DP_new[df_DP_new.cluster_no == value].Time_in_Drought_Before)+29,0),round(np.nanmean(df_DP_new[df_DP_new.cluster_no == value].Time_in_Pluvial_After)+29,0)] for key, value in clusters.items()} 
cluster_averages_PD = {key: [round(np.nanmean(df_PD_new[df_PD_new.cluster_no == value].Time_in_Pluvial_Before)+29,0),round(np.nanmean(df_PD_new[df_PD_new.cluster_no == value].Time_in_Drought_After)+29,0)] for key, value in clusters.items()} 

cluster_averages_droughts = {key: [round(np.nanmean(df_droughts_new[df_droughts_new.cluster_no == value].Total_Time),3)] for key, value in clusters.items()} 
cluster_averages_pluvials = {key: [round(np.nanmean(df_pluvials_new[df_pluvials_new.cluster_no == value].Total_Time),3)] for key, value in clusters.items()} 

cluster_averages_droughts = {key: [round(np.nanmean(df_droughts_new[df_droughts_new.cluster_no == value].Total_Time)+29,0)] for key, value in clusters.items()} 
cluster_averages_pluvials = {key: [round(np.nanmean(df_pluvials_new[df_pluvials_new.cluster_no == value].Total_Time)+29,0)] for key, value in clusters.items()} 


#########################################################################################
#Season
#########################################################################################
seasons = {'winter':1, 'spring':2, 'summer':3, 'fall':4}

seasonal_number_DP = {key: len(df_DP_new[df_DP_new.season == value].reset_index(drop=True)) for key, value in seasons.items()} 
seasonal_number_PD = {key: len(df_PD_new[df_PD_new.season == value].reset_index(drop=True)) for key, value in seasons.items()} 

seasonal_averages_DP = {key: [round(np.nanmean(df_DP_new[df_DP_new.season == value].Time_in_Drought_Before),3),round(np.nanmean(df_DP_new[df_DP_new.season == value].Time_in_Pluvial_After),3)] for key, value in seasons.items()} 
seasonal_averages_PD = {key: [round(np.nanmean(df_PD_new[df_PD_new.season == value].Time_in_Pluvial_Before),3),round(np.nanmean(df_PD_new[df_PD_new.season == value].Time_in_Drought_After),3)] for key, value in seasons.items()} 

#########################################################################################
#Month
#########################################################################################
months = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

monthly_number_DP = {key: len(df_DP_new[df_DP_new.month == value].reset_index(drop=True)) for key, value in months.items()} 
monthly_number_PD = {key: len(df_PD_new[df_PD_new.month == value].reset_index(drop=True)) for key, value in months.items()} 

monthly_averages_DP = {key: [round(np.nanmean(df_DP_new[df_DP_new.month == value].Time_in_Drought_Before),3),round(np.nanmean(df_DP_new[df_DP_new.month == value].Time_in_Pluvial_After),3)] for key, value in months.items()} 
monthly_averages_PD = {key: [round(np.nanmean(df_PD_new[df_PD_new.month == value].Time_in_Pluvial_Before),3),round(np.nanmean(df_PD_new[df_PD_new.month == value].Time_in_Drought_After),3)] for key, value in months.items()} 

#########################################################################################
#Are the averages significantly different from each other?
#########################################################################################
def test_significance(df):
	
	significant_values = []
	
	for i in range(len(df)):
		test_val = df[i]
		rest = np.delete(df, i)
		t_stat, p_val = ttest_1samp(rest, test_val)
		
		if p_val < 0.05:
			significant_values.append([i+1,test_val])
		else:
			pass
	
	return significant_values


significance_dp = {'cluster_averages_before':  test_significance([v[0] for v in cluster_averages_DP.values()]), 
					'cluster_averages_after':  test_significance([v[1] for v in cluster_averages_DP.values()]), 
					
					'seasonal_averages_before':  test_significance([v[0] for v in seasonal_averages_DP.values()]), 
					'seasonal_averages_after':  test_significance([v[1] for v in seasonal_averages_DP.values()]), 
					
					'monthly_averages_before':   test_significance([v[0] for v in monthly_averages_DP.values()]), 
					'monthly_averages_after':   test_significance([v[1] for v in monthly_averages_DP.values()]), 
					
					'intensity_averages_before': test_significance(intensity_averages_DP.Time_Before),
					'intensity_averages_after': test_significance(intensity_averages_DP.Time_After)
					}

significance_pd = {'cluster_averages_before':  test_significance([v[0] for v in cluster_averages_PD.values()]), 
					'cluster_averages_after':  test_significance([v[1] for v in cluster_averages_PD.values()]), 
					
					'seasonal_averages_before':  test_significance([v[0] for v in seasonal_averages_PD.values()]), 
					'seasonal_averages_after':  test_significance([v[1] for v in seasonal_averages_PD.values()]), 
					
					'monthly_averages_before':   test_significance([v[0] for v in monthly_averages_PD.values()]), 
					'monthly_averages_after':   test_significance([v[1] for v in monthly_averages_PD.values()]), 
					
					'intensity_averages_before': test_significance(intensity_averages_PD.Time_Before),
					'intensity_averages_after': test_significance(intensity_averages_PD.Time_After)
					}

significance_droughts = {'cluster_averages':  test_significance([v[0] for v in cluster_averages_droughts.values()]), 
					}

significance_pluvials = {'cluster_averages':  test_significance([v[0] for v in cluster_averages_pluvials.values()]), 
					}


#########################################################################################
#Plot (Scatter Plots) - small (Figure 11 in manuscript)
#########################################################################################
df_DP_small = df_DP_new[df_DP_new.Time_in_Pluvial_After <=100].reset_index(drop=True) 
df_DP_small = df_DP_small[df_DP_small.Time_in_Drought_Before <=100].reset_index(drop=True) 

df_PD_small = df_PD_new[df_PD_new.Time_in_Pluvial_Before <=100].reset_index(drop=True) 
df_PD_small = df_PD_small[df_PD_small.Time_in_Drought_After <=100].reset_index(drop=True) 

fig = plt.figure(figsize = (12,12), dpi = 300, tight_layout =True)

#########################################################################################
#Cluster
#########################################################################################
colors_DP = df_DP_small.cluster_no
colors_PD = df_PD_small.cluster_no
vmin=1
vmax=8

cmap = ListedColormap(['red', 'green', 'purple', 'saddlebrown', 'blue', 'black', 'orange'])
colors = cycle(['red', 'green', 'purple', 'saddlebrown', 'blue', 'black', 'orange'])

#Drought-to-Pluvial
ax1 = fig.add_subplot(221)

plt.scatter(df_DP_small.Time_in_Drought_Before+29, df_DP_small.Time_in_Pluvial_After+29, s=1, c=colors_DP,cmap=cmap,vmin=vmin,vmax=vmax, alpha = 0.8)

for key,value in cluster_averages_DP.items():
	plt.plot(value[0]+29, value[1]+29, '*', color=next(colors), markersize=15, markeredgecolor='k', label=key)

plt.xticks(np.arange(20, 150, 20))
ax1.set_xticklabels(['20','40','60','80','100','120','140'], fontsize=7)
plt.yticks(np.arange(20, 150, 20))
ax1.set_yticklabels(['20','40','60','80','100','120','140'], fontsize=7)

ax1.set_xlabel('Time in Drought Before (Days)', fontsize = 10)
ax1.set_ylabel('Time in Pluvial After (Days)', fontsize = 10)	

# Add colorbar with custom ticks
cbar = plt.colorbar(pad=0.05)
cbar.set_ticks(np.arange(1.5,8.5,1))
cbar.set_ticklabels(['Cluster 1','Cluster 2',' Cluster 3','Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
cbar.ax.set_title('Cluster',fontsize=7)
cbar.ax.tick_params(labelsize=7) 	

plt.title('a) Clusters', loc = "left",fontsize=10)
plt.title("Drought-to-Pluvial", loc = 'right',fontsize=10)


#Pluvial-to-Drought
ax2 = fig.add_subplot(222)

plt.scatter(df_PD_small.Time_in_Pluvial_Before+29, df_PD_small.Time_in_Drought_After+29, s=1, c=colors_PD,cmap=cmap,vmin=vmin,vmax=vmax,alpha = 0.8)

for key,value in cluster_averages_PD.items():
	plt.plot(value[0]+29, value[1]+29, '*', color=next(colors), markersize=15, markeredgecolor='k', label=key)

plt.xticks(np.arange(20, 150, 20))
ax2.set_xticklabels(['20','40','60','80','100','120','140'], fontsize=7)
plt.yticks(np.arange(20, 150, 20))
ax2.set_yticklabels(['20','40','60','80','100','120','140'], fontsize=7)

ax2.set_xlabel('Time in Pluvial Before (Days)', fontsize = 10)
ax2.set_ylabel('Time in Drought After (Days)', fontsize = 10)

# Add colorbar with custom ticks
cbar = plt.colorbar(pad=0.05)
cbar.set_ticks(np.arange(1.5,8.5,1))
cbar.set_ticklabels(['Cluster 1','Cluster 2',' Cluster 3','Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
cbar.ax.set_title('Cluster',fontsize=7)
cbar.ax.tick_params(labelsize=7) 	

plt.title('b) Clusters', loc = "left",fontsize=10)
plt.title("Pluvial-to-Drought", loc = 'right',fontsize=10)


#########################################################################################
#Intensity
#########################################################################################
intensity_dp_small = intensity_dp[intensity_dp.Time_Before <=100].reset_index(drop=True) 
intensity_dp_small = intensity_dp_small[intensity_dp_small.Time_After <=100].reset_index(drop=True) 

intensity_pd_small = intensity_pd[intensity_pd.Time_Before <=100].reset_index(drop=True) 
intensity_pd_small = intensity_pd_small[intensity_pd_small.Time_After <=100].reset_index(drop=True) 

colors_DP = intensity_dp_small.Avg_SPI_Change
colors_PD = intensity_pd_small.Avg_SPI_Change
vmin=1
vmax=4.5

cmap = plt.colormaps['gist_rainbow']
bounds = [1, 2, 3, 4, 4.5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Get only the colors that correspond to the bounds
colors = cycle([cmap(norm(b)) for b in bounds])  # Cycle through discrete colormap values


#Drought-to-Pluvial
ax3 = fig.add_subplot(223)

plt.scatter(intensity_dp_small.Time_Before+29, intensity_dp_small.Time_After+29, s=1, c=colors_DP,cmap=cmap,vmin=vmin,vmax=vmax, alpha = 0.8)

for row in range(0, len(intensity_averages_DP)):
	plt.plot(intensity_averages_DP.Time_Before[row]+29, intensity_averages_DP.Time_After[row]+29, '*', color=next(colors), markersize=15, markeredgecolor='k',  label = intensity_averages_DP.Intensity_bins)
	
plt.xticks(np.arange(20, 150, 20))
ax3.set_xticklabels(['20','40','60','80','100','120','140'], fontsize=7)
plt.yticks(np.arange(20, 150, 20))
ax3.set_yticklabels(['20','40','60','80','100','120','140'], fontsize=7)

ax3.set_xlabel('Time in Drought Before (Days)', fontsize = 10)
ax3.set_ylabel('Time in Pluvial After (Days)', fontsize = 10)	

# Add colorbar with custom ticks
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax3, pad=0.05)

cbar.set_ticks(bounds)
cbar.ax.set_title('SPI \nChange',fontsize=7)
cbar.ax.tick_params(labelsize=7) 	

plt.title('c) Intensity', loc = "left",fontsize=10)
plt.title("Drought-to-Pluvial", loc = 'right',fontsize=10)

cmap = plt.colormaps['gist_rainbow']
bounds = [1, 2, 3, 4, 4.5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Get only the colors that correspond to the bounds
colors = cycle([cmap(norm(b)) for b in bounds])  # Cycle through discrete colormap values


#Pluvial-to-Drought
ax4 = fig.add_subplot(224)

plt.scatter(intensity_pd_small.Time_Before+29, intensity_pd_small.Time_After+29, s=1, c=colors_PD,cmap=cmap,vmin=vmin,vmax=vmax,alpha = 0.8)

for row in range(0, len(intensity_averages_PD)):
	plt.plot(intensity_averages_PD.Time_Before[row]+29, intensity_averages_PD.Time_After[row]+29, '*', color=next(colors), markersize=15, markeredgecolor='k',  label = intensity_averages_PD.Intensity_bins)

plt.xticks(np.arange(20, 150, 20))
ax4.set_xticklabels(['20','40','60','80','100','120','140'], fontsize=7)
plt.yticks(np.arange(20, 150, 20))
ax4.set_yticklabels(['20','40','60','80','100','120','140'], fontsize=7)

ax4.set_xlabel('Time in Pluvial Before (Days)', fontsize = 10)
ax4.set_ylabel('Time in Drought After (Days)', fontsize = 10)

# Add colorbar with custom ticks
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax4, pad=0.05)

cbar.set_ticks(bounds)
cbar.ax.set_title('SPI \nChange',fontsize=7)
cbar.ax.tick_params(labelsize=7) 	

plt.title('d) Intensity', loc = "left",fontsize=10)
plt.title("Pluvial-to-Drought", loc = 'right',fontsize=10)


plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_length_cluster_intensity_small.eps', bbox_inches = 'tight', pad_inches = 0.1) 

#########################################################################################
#Plot (Scatter Plots) - large (Supplementary Figure 7)
#########################################################################################
fig = plt.figure(figsize = (12,12), dpi = 300, tight_layout =True)

#########################################################################################
#Cluster
#########################################################################################
colors_DP = df_DP_new.cluster_no
colors_PD = df_PD_new.cluster_no
vmin=1
vmax=8

cmap = ListedColormap(['red', 'green', 'purple', 'saddlebrown', 'blue', 'black', 'orange'])
colors = cycle(['red', 'green', 'purple', 'saddlebrown', 'blue', 'black', 'orange'])

#Drought-to-Pluvial
ax1 = fig.add_subplot(221)

plt.scatter(df_DP_new.Time_in_Drought_Before+29, df_DP_new.Time_in_Pluvial_After+29, s=1, c=colors_DP,cmap=cmap,vmin=vmin,vmax=vmax)

for key,value in cluster_averages_DP.items():
	plt.plot(value[0]+29, value[1]+29, '*', color=next(colors), markersize=15, markeredgecolor='k', label=key)

plt.xticks(np.arange(20, 350, 20))
ax1.set_xticklabels(['20','40','60','80','100','120','140','160','180','200','220','240','260','280','300','320','340'], fontsize=7)
plt.yticks(np.arange(20, 350, 20))
ax1.set_yticklabels(['20','40','60','80','100','120','140','160','180','200','220','240','260','280','300','320','340'], fontsize=7)

ax1.set_xlabel('Time in Drought Before (Days)', fontsize = 10)
ax1.set_ylabel('Time in Pluvial After (Days)', fontsize = 10)	

# Add colorbar with custom ticks
cbar = plt.colorbar(pad=0.05)
cbar.set_ticks(np.arange(1.5,8.5,1))
cbar.set_ticklabels(['Cluster 1','Cluster 2',' Cluster 3','Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
cbar.ax.set_title('Cluster',fontsize=7)
cbar.ax.tick_params(labelsize=7) 	

plt.title('a) Clusters', loc = "left",fontsize=10)
plt.title("Drought-to-Pluvial", loc = 'right',fontsize=10)


#Pluvial-to-Drought
ax2 = fig.add_subplot(222)

plt.scatter(df_PD_new.Time_in_Pluvial_Before+29, df_PD_new.Time_in_Drought_After+29, s=1, c=colors_PD,cmap=cmap,vmin=vmin,vmax=vmax)

for key,value in cluster_averages_PD.items():
	plt.plot(value[0]+29, value[1]+29, '*', color=next(colors), markersize=15, markeredgecolor='k', label=key)

plt.xticks(np.arange(20, 350, 20))
ax2.set_xticklabels(['20','40','60','80','100','120','140','160','180','200','220','240','260','280','300','320','340'], fontsize=7)
plt.yticks(np.arange(20, 350, 20))
ax2.set_yticklabels(['20','40','60','80','100','120','140','160','180','200','220','240','260','280','300','320','340'], fontsize=7)

ax2.set_xlabel('Time in Pluvial Before (Days)', fontsize = 10)
ax2.set_ylabel('Time in Drought After (Days)', fontsize = 10)

# Add colorbar with custom ticks
cbar = plt.colorbar(pad=0.05)
cbar.set_ticks(np.arange(1.5,8.5,1))
cbar.set_ticklabels(['Cluster 1','Cluster 2',' Cluster 3','Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
cbar.ax.set_title('Cluster',fontsize=7)
cbar.ax.tick_params(labelsize=7) 	

plt.title('b) Clusters', loc = "left",fontsize=10)
plt.title("Pluvial-to-Drought", loc = 'right',fontsize=10)


#########################################################################################
#Intensity
#########################################################################################
colors_DP = intensity_dp.Avg_SPI_Change
colors_PD = intensity_pd.Avg_SPI_Change
vmin=1
vmax=4.5

cmap = plt.colormaps['gist_rainbow']
bounds = [1, 2, 3, 4, 4.5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Get only the colors that correspond to the bounds
colors = cycle([cmap(norm(b)) for b in bounds])  # Cycle through discrete colormap values


#Drought-to-Pluvial
ax3 = fig.add_subplot(223)

plt.scatter(intensity_dp.Time_Before+29, intensity_dp.Time_After+29, s=1, c=colors_DP,cmap=cmap,vmin=vmin,vmax=vmax)

for row in range(0, len(intensity_averages_DP)):
	plt.plot(intensity_averages_DP.Time_Before[row]+29, intensity_averages_DP.Time_After[row]+29, '*', color=next(colors), markersize=15, markeredgecolor='k',  label = intensity_averages_DP.Intensity_bins)

plt.xticks(np.arange(20, 350, 20))
ax3.set_xticklabels(['20','40','60','80','100','120','140','160','180','200','220','240','260','280','300','320','340'], fontsize=7)
plt.yticks(np.arange(20, 350, 20))
ax3.set_yticklabels(['20','40','60','80','100','120','140','160','180','200','220','240','260','280','300','320','340'], fontsize=7)

ax3.set_xlabel('Time in Drought Before (Days)', fontsize = 10)
ax3.set_ylabel('Time in Pluvial After (Days)', fontsize = 10)	

# Add colorbar with custom ticks
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax3, pad=0.05)

cbar.set_ticks(bounds)
cbar.ax.set_title('SPI \nChange',fontsize=7)
cbar.ax.tick_params(labelsize=7) 	

plt.title('c) Intensity', loc = "left",fontsize=10)
plt.title("Drought-to-Pluvial", loc = 'right',fontsize=10)

cmap = plt.colormaps['gist_rainbow']
bounds = [1, 2, 3, 4, 4.5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Get only the colors that correspond to the bounds
colors = cycle([cmap(norm(b)) for b in bounds])  # Cycle through discrete colormap values


#Pluvial-to-Drought
ax4 = fig.add_subplot(224)

plt.scatter(intensity_pd.Time_Before+29, intensity_pd.Time_After+29, s=1, c=colors_PD,cmap=cmap,vmin=vmin,vmax=vmax)

for row in range(0, len(intensity_averages_PD)):
	plt.plot(intensity_averages_PD.Time_Before[row]+29, intensity_averages_PD.Time_After[row]+29, '*', color=next(colors), markersize=15, markeredgecolor='k',  label = intensity_averages_PD.Intensity_bins)

plt.xticks(np.arange(20, 350, 20))
ax4.set_xticklabels(['20','40','60','80','100','120','140','160','180','200','220','240','260','280','300','320','340'], fontsize=7)
plt.yticks(np.arange(20, 350, 20))
ax4.set_yticklabels(['20','40','60','80','100','120','140','160','180','200','220','240','260','280','300','320','340'], fontsize=7)

ax4.set_xlabel('Time in Pluvial Before (Days)', fontsize = 10)
ax4.set_ylabel('Time in Drought After (Days)', fontsize = 10)

# Add colorbar with custom ticks
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax4, pad=0.05)

cbar.set_ticks(bounds)
cbar.ax.set_title('SPI \nChange',fontsize=7)
cbar.ax.tick_params(labelsize=7) 	

plt.title('d) Intensity', loc = "left",fontsize=10)
plt.title("Pluvial-to-Drought", loc = 'right',fontsize=10)


plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_length_cluster_intensity.eps', bbox_inches = 'tight', pad_inches = 0.1) 
