#########################################################################################
## A script to calculate how far the center point of the polygon is moving during each 
## event (Event Propagation) and how the distance between the center point of the polygon 
## and the boundary is changing during each event (Event Growth)
## Bryony Louise Puxley
## Last Edited: Monday, August 11th, 2025 
## Input: Independent event files of Drought-to-Pluvial, Pluvial-to-Drought, Drought only,
## and Pluvial only events.
## Output: A PNG of a scatter plot of centroid propagation vs event growth (Figure 12)
#########################################################################################
#Import Required Modules
#########################################################################################
import numpy as np
import pandas as pd
import scipy.stats as scs
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from itertools import cycle
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import shapely.wkt
from shapely.geometry import Polygon, Point

#########################################################################################
#Import Functions
#########################################################################################
import functions

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

#########################################################################################
#Polygons
#########################################################################################
polygons_DP = [shapely.wkt.loads(i) for i in df_DP.geometry] #convert from nasty string of lat,lons to geometry object
polygons_PD = [shapely.wkt.loads(i) for i in df_PD.geometry] #convert from nasty string of lat,lons to geometry object
polygons_droughts = [shapely.wkt.loads(i) for i in df_droughts.geometry] #convert from nasty string of lat,lons to geometry object
polygons_pluvials = [shapely.wkt.loads(i) for i in df_pluvials.geometry] #convert from nasty string of lat,lons to geometry object

df_DP['polygon'] = polygons_DP
df_PD['polygon'] = polygons_PD
df_droughts['polygon'] = polygons_droughts
df_pluvials['polygon'] = polygons_pluvials

#Find the center of each polygon
df_DP['centroids'] = [poly.centroid for poly in polygons_DP] #get centroid of polygons
df_PD['centroids'] = [poly.centroid for poly in polygons_PD] #get centroid of polygons
df_droughts['centroids'] = [poly.centroid for poly in polygons_droughts] #get centroid of polygons
df_pluvials['centroids'] = [poly.centroid for poly in polygons_pluvials] #get centroid of polygons

#Find the boundary/exterior points of each polygon
df_DP['boundary_points'] = [list(poly.exterior.coords) for poly in polygons_DP]
df_PD['boundary_points'] = [list(poly.exterior.coords) for poly in polygons_PD]
df_droughts['boundary_points'] = [list(poly.exterior.coords) for poly in polygons_droughts]
df_pluvials['boundary_points'] = [list(poly.exterior.coords) for poly in polygons_pluvials]


#########################################################################################
#Find how far the center point of the polygon is moving during each event (Event Propagation)
#########################################################################################

def centroid_propagation(df):
	dist_deg = []
	dist_km = []
	event_len = []
	cluster_no = []
	for i in tqdm(range(0,np.nanmax(df.Event_No))):
		#subset database to individual events
		subset_ind = np.where((df.Event_No == i+1))[0] 
		subset =  df.iloc[subset_ind]
		event_len.append(len(subset))
		cluster_no.append( np.nanmax(subset.cluster_no))

		dist_deg.append((subset.centroids.iloc[-1].distance(subset.centroids.iloc[0]))/len(subset)) #distance in the same units as input coordinates between first and last point

		dist_km.append((functions.haversine(subset.centroids.iloc[0].x, subset.centroids.iloc[0].y, subset.centroids.iloc[-1].x, subset.centroids.iloc[-1].y))/len(subset))

	dist = pd.DataFrame({'cluster':cluster_no, 'event_length': event_len, 'Distance_deg': dist_deg, 'Distance_km':dist_km})
	
	return dist
	
centroid_propagation_DP = centroid_propagation(df_DP)
centroid_propagation_PD = centroid_propagation(df_PD)
centroid_propagation_droughts = centroid_propagation(df_droughts)
centroid_propagation_pluvials = centroid_propagation(df_pluvials)

#########################################################################################
#Find how the distance between the center point of the polygon and the boundary is changing 
#during each event (Event Growth)
#########################################################################################

def event_growth(df):
	dist_km = []
	event_len = []
	cluster_no = []
	for i in tqdm(range(0,np.nanmax(df.Event_No))):
		#subset database to individual events
		subset_ind = np.where((df.Event_No == i+1))[0] 
		subset =  df.iloc[subset_ind]
		event_len.append(len(subset))
		cluster_no.append(np.nanmax(subset.cluster_no))

		avg_distances_deg = []
		avg_distances_km = []

		#calculate the average distance between each center point and the exterior coordinates for each row in the subset
		for row in range(0,len(subset)):
			avg_distances_deg.append(np.nanmean([subset.centroids.iloc[row].distance(Point(p)) for p in subset.boundary_points.iloc[row]]))
			avg_distances_km.append(np.nanmean([functions.haversine(subset.centroids.iloc[row].x, subset.centroids.iloc[row].y, lat, lon) for lon, lat in subset.boundary_points.iloc[row]])) 

		#Calculate the difference between the smallest distance and the largest distance
		index_difference = abs(np.argmax(avg_distances_km) -  np.argmin(avg_distances_km))
		dist_km.append((np.nanmax(avg_distances_km) - np.nanmin(avg_distances_km))/index_difference)
	
	dist = pd.DataFrame({'cluster':cluster_no, 'event_length': event_len, 'Distance_km':dist_km})	

	return dist

event_growth_DP = event_growth(df_DP)
event_growth_PD = event_growth(df_PD)
event_growth_droughts = event_growth(df_droughts)
event_growth_pluvials = event_growth(df_pluvials)

#########################################################################################
#Filter out the events where the event lengths are only 1 day
#########################################################################################
centroid_propagation_DP_filtered = centroid_propagation_DP[centroid_propagation_DP["event_length"] > 1].reset_index(drop=True)
event_growth_DP_filtered = event_growth_DP[event_growth_DP["event_length"] > 1].reset_index(drop=True)

centroid_propagation_PD_filtered = centroid_propagation_PD[centroid_propagation_PD["event_length"] > 1].reset_index(drop=True)
event_growth_PD_filtered = event_growth_PD[event_growth_PD["event_length"] > 1].reset_index(drop=True)

centroid_propagation_droughts_filtered = centroid_propagation_droughts[centroid_propagation_droughts["event_length"] > 1].reset_index(drop=True)
event_growth_droughts_filtered = event_growth_droughts[event_growth_droughts["event_length"] > 1].reset_index(drop=True)

centroid_propagation_pluvials_filtered = centroid_propagation_pluvials[centroid_propagation_pluvials["event_length"] > 1].reset_index(drop=True)
event_growth_pluvials_filtered = event_growth_pluvials[event_growth_pluvials["event_length"] > 1].reset_index(drop=True)


#########################################################################################
#Calculate Cluster Averages and Length Bins
#########################################################################################
#Clusters
clusters = {'Cluster_1':1,'Cluster_2':2,'Cluster_3':3,'Cluster_4':4,'Cluster_5':5,'Cluster_6':6,'Cluster_7':7}

cluster_averages_DP = {key: [round(np.nanmean(centroid_propagation_DP_filtered[centroid_propagation_DP_filtered.cluster == value].Distance_km),2), round(np.nanmean(event_growth_DP_filtered[event_growth_DP_filtered.cluster == value].Distance_km),2)] for key, value in clusters.items()} 
cluster_averages_PD = {key: [round(np.nanmean(centroid_propagation_PD_filtered[centroid_propagation_PD_filtered.cluster == value].Distance_km),2), round(np.nanmean(event_growth_PD_filtered[event_growth_PD_filtered.cluster == value].Distance_km),2)] for key, value in clusters.items()} 
cluster_averages_droughts = {key: [round(np.nanmean(centroid_propagation_droughts_filtered[centroid_propagation_droughts_filtered.cluster == value].Distance_km),2), round(np.nanmean(event_growth_droughts_filtered[event_growth_droughts_filtered.cluster == value].Distance_km),2)] for key, value in clusters.items()} 
cluster_averages_pluvials = {key: [round(np.nanmean(centroid_propagation_pluvials_filtered[centroid_propagation_pluvials_filtered.cluster == value].Distance_km),2), round(np.nanmean(event_growth_pluvials_filtered[event_growth_pluvials_filtered.cluster == value].Distance_km),2)] for key, value in clusters.items()} 


#Length
bins = [1, 5, 10, 15, 23]  # Adjust as needed
bin_labels = [f"{bins[i]} - {bins[i+1]}" for i in range(len(bins)-1)]

centroid_propagation_DP_filtered['event_length_bins'] = pd.cut(centroid_propagation_DP_filtered['event_length'], bins=bins, right=False)
centroid_propagation_PD_filtered['event_length_bins'] = pd.cut(centroid_propagation_PD_filtered['event_length'], bins=bins, right=False)
centroid_propagation_droughts_filtered['event_length_bins'] = pd.cut(centroid_propagation_droughts_filtered['event_length'], bins=bins, right=False)
centroid_propagation_pluvials_filtered['event_length_bins'] = pd.cut(centroid_propagation_pluvials_filtered['event_length'], bins=bins, right=False)


event_growth_DP_filtered['event_length_bins'] = pd.cut(event_growth_DP_filtered['event_length'], bins=bins, right=False)
event_growth_PD_filtered['event_length_bins'] = pd.cut(event_growth_PD_filtered['event_length'], bins=bins, right=False)
event_growth_droughts_filtered['event_length_bins'] = pd.cut(event_growth_droughts_filtered['event_length'], bins=bins, right=False)
event_growth_pluvials_filtered['event_length_bins'] = pd.cut(event_growth_pluvials_filtered['event_length'], bins=bins, right=False)


length_averages_DP = pd.DataFrame({'Bins': bin_labels, 'Propagation': centroid_propagation_DP_filtered.groupby('event_length_bins')['Distance_km'].mean().reset_index(drop=True), 'Growth': event_growth_DP_filtered.groupby('event_length_bins')['Distance_km'].mean().reset_index(drop=True)})
length_averages_PD = pd.DataFrame({'Bins': bin_labels, 'Propagation': centroid_propagation_PD_filtered.groupby('event_length_bins')['Distance_km'].mean().reset_index(drop=True), 'Growth': event_growth_PD_filtered.groupby('event_length_bins')['Distance_km'].mean().reset_index(drop=True)})
length_averages_droughts = pd.DataFrame({'Bins': bin_labels, 'Propagation': centroid_propagation_droughts_filtered.groupby('event_length_bins')['Distance_km'].mean().reset_index(drop=True), 'Growth': event_growth_droughts_filtered.groupby('event_length_bins')['Distance_km'].mean().reset_index(drop=True)})
length_averages_pluvials = pd.DataFrame({'Bins': bin_labels, 'Propagation': centroid_propagation_pluvials_filtered.groupby('event_length_bins')['Distance_km'].mean().reset_index(drop=True), 'Growth': event_growth_pluvials_filtered.groupby('event_length_bins')['Distance_km'].mean().reset_index(drop=True)})

averages = {'Drought-to-Pluvial': {'Propagation': np.nanmean(centroid_propagation_DP_filtered.Distance_km), 
									'Growth': np.nanmean(event_growth_DP_filtered.Distance_km)}, 
			'Pluvial-to-Drought': {'Propagation': np.nanmean(centroid_propagation_PD_filtered.Distance_km), 
									'Growth': np.nanmean(event_growth_PD_filtered.Distance_km)}, 
			'Drought': {'Propagation': np.nanmean(centroid_propagation_droughts_filtered.Distance_km), 
									'Growth': np.nanmean(event_growth_droughts_filtered.Distance_km)}, 
			'Pluvial': {'Propagation': np.nanmean(centroid_propagation_pluvials_filtered.Distance_km), 
									'Growth': np.nanmean(event_growth_pluvials_filtered.Distance_km)}, 
}


#########################################################################################
#Calculate the line of best fit
#########################################################################################
res_DP = scs.linregress(centroid_propagation_DP_filtered.Distance_km, event_growth_DP_filtered.Distance_km)
res_PD = scs.linregress(centroid_propagation_PD_filtered.Distance_km, event_growth_PD_filtered.Distance_km)
res_droughts = scs.linregress(centroid_propagation_droughts_filtered.Distance_km, event_growth_droughts_filtered.Distance_km)
res_pluvials = scs.linregress(centroid_propagation_pluvials_filtered.Distance_km, event_growth_pluvials_filtered.Distance_km)


x = y = list(range(0, 400, 100))

#########################################################################################
#Normalized scatter plot of centroid propagation vs event growth (large)
#########################################################################################
fig = plt.figure(figsize = (12,12), dpi = 300, tight_layout =True)

x = y = list(range(0, 1001, 100))
#Drought-to-Pluvial
ax1 = fig.add_subplot(221)

plt.scatter(centroid_propagation_DP_filtered.Distance_km, event_growth_DP_filtered.Distance_km, s=1, c='mediumspringgreen')
plt.plot(centroid_propagation_DP_filtered.Distance_km, res_DP.intercept + res_DP.slope*centroid_propagation_DP_filtered.Distance_km, 'blue', label=r'Fitted Line, Slope = {0}'.format(round(res_DP.slope,3)))
plt.plot(x, y, color='k',label = 'Line of Equality, Slope = 1.0')
plt.plot(averages['Drought-to-Pluvial']['Propagation'], averages['Drought-to-Pluvial']['Growth'], '*', color='blue', markersize=15, markeredgecolor='k')


plt.xticks(np.arange(0, 1100, 100))
ax1.set_xticklabels(np.arange(0, 1100, 100), fontsize=7)
plt.yticks(np.arange(0, 1100, 100))
ax1.set_yticklabels(np.arange(0, 1100, 100), fontsize=7)

ax1.set_xlabel('Centroid Propagation (km)', fontsize = 10)
ax1.set_ylabel('Event Growth (km)', fontsize = 10)	

plt.legend(loc='upper left', fontsize=7, framealpha=1)

plt.title("a) Drought-to-Pluvial Events", fontsize=10, loc='left')

#Pluvial-to-Drought
ax2 = fig.add_subplot(222)
plt.scatter(centroid_propagation_PD_filtered.Distance_km, event_growth_PD_filtered.Distance_km, s=1, c='mediumspringgreen')
plt.plot(centroid_propagation_PD_filtered.Distance_km, res_PD.intercept + res_PD.slope*centroid_propagation_PD_filtered.Distance_km, 'blue', label=r'Fitted Line, Slope = {0}'.format(round(res_PD.slope,3)))
plt.plot(x, y, color='k',label = 'Line of Equality, Slope = 1.0')
plt.plot(averages['Pluvial-to-Drought']['Propagation'], averages['Pluvial-to-Drought']['Growth'], '*', color='blue', markersize=15, markeredgecolor='k')

plt.xticks(np.arange(0, 1100, 100))
ax2.set_xticklabels(np.arange(0, 1100, 100), fontsize=7)
plt.yticks(np.arange(0, 1100, 100))
ax2.set_yticklabels(np.arange(0, 1100, 100), fontsize=7)

ax2.set_xlabel('Centroid Propagation (km)', fontsize = 10)
ax2.set_ylabel('Event Growth (km)', fontsize = 10)	

plt.legend(loc='upper left', fontsize=7, framealpha=1) 	

plt.title("b) Pluvial-to-Drought Events", fontsize=10, loc ='left')

#Droughts
ax3 = fig.add_subplot(223)

plt.scatter(centroid_propagation_droughts_filtered.Distance_km, event_growth_droughts_filtered.Distance_km, s=1, c='mediumspringgreen')
plt.plot(centroid_propagation_droughts_filtered.Distance_km, res_droughts.intercept + res_droughts.slope*centroid_propagation_droughts_filtered.Distance_km, 'blue', label=r'Fitted Line, Slope = {0}'.format(round(res_droughts.slope,3)))
plt.plot(x, y, color='k',label = 'Line of Equality, Slope = 1.0')
plt.plot(averages['Drought']['Propagation'], averages['Drought']['Growth'], '*', color='blue', markersize=15, markeredgecolor='k')

plt.xticks(np.arange(0, 1100, 100))
ax3.set_xticklabels(np.arange(0, 1100, 100), fontsize=7)
plt.yticks(np.arange(0, 1100, 100))
ax3.set_yticklabels(np.arange(0, 1100, 100), fontsize=7)

ax3.set_xlabel('Centroid Propagation (km)', fontsize = 10)
ax3.set_ylabel('Event Growth (km)', fontsize = 10)	

plt.legend(loc='upper left', fontsize=7, framealpha=1)

plt.title("c) Drought Events", fontsize=10, loc='left')

#Pluvials
ax4 = fig.add_subplot(224)
plt.scatter(centroid_propagation_pluvials_filtered.Distance_km, event_growth_pluvials_filtered.Distance_km, s=1, c='mediumspringgreen')
plt.plot(centroid_propagation_pluvials_filtered.Distance_km, res_pluvials.intercept + res_pluvials.slope*centroid_propagation_pluvials_filtered.Distance_km, 'blue', label=r'Fitted Line, Slope = {0}'.format(round(res_pluvials.slope,3)))
plt.plot(x, y, color='k',label = 'Line of Equality, Slope = 1.0')
plt.plot(averages['Pluvial']['Propagation'], averages['Pluvial']['Growth'], '*', color='blue', markersize=15, markeredgecolor='k')

plt.xticks(np.arange(0, 1100, 100))
ax4.set_xticklabels(np.arange(0, 1100, 100), fontsize=7)
plt.yticks(np.arange(0, 1100, 100))
ax4.set_yticklabels(np.arange(0, 1100, 100), fontsize=7)

ax4.set_xlabel('Centroid Propagation (km)', fontsize = 10)
ax4.set_ylabel('Event Growth (km)', fontsize = 10)	

plt.legend(loc='upper left', fontsize=7, framealpha=1) 	

plt.title("d) Pluvial Events", fontsize=10, loc ='left')


plt.tight_layout()

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_propagation_vs_growth_cluster_length_normalized.png', bbox_inches = 'tight', pad_inches = 0.1)    


#########################################################################################
# Normalized scatter plot of centroid propagation vs event growth (Small)
#########################################################################################
fig = plt.figure(figsize = (12,12), dpi = 300, tight_layout =True)

x = y = list(range(0, 400, 100))
#Drought-to-Pluvial
ax1 = fig.add_subplot(141)

plt.scatter(centroid_propagation_DP_filtered.Distance_km, event_growth_DP_filtered.Distance_km, s=1, c='mediumspringgreen')
plt.plot(centroid_propagation_DP_filtered.Distance_km, res_DP.intercept + res_DP.slope*centroid_propagation_DP_filtered.Distance_km, 'blue', label=r'Fitted Line, Slope = {0}'.format(round(res_DP.slope,3)))
plt.plot(x, y, color='k',label = 'Line of Equality, Slope = 1.0')
plt.plot(averages['Drought-to-Pluvial']['Propagation'], averages['Drought-to-Pluvial']['Growth'], '*', color='blue', markersize=15, markeredgecolor='k')


plt.xticks(np.arange(0, 400, 100))
ax1.set_xticklabels(np.arange(0, 400, 100), fontsize=7)
plt.yticks(np.arange(0, 1100, 100))
ax1.set_yticklabels(np.arange(0, 1100, 100), fontsize=7)

ax1.set_xlabel('Centroid Propagation (km)', fontsize = 10)
ax1.set_ylabel('Event Growth (km)', fontsize = 10)	

plt.legend(loc='upper left', fontsize=7, framealpha=1)

plt.title("a) Drought-to-Pluvial Events", fontsize=10, loc='left')

#Pluvial-to-Drought
ax2 = fig.add_subplot(142)
plt.scatter(centroid_propagation_PD_filtered.Distance_km, event_growth_PD_filtered.Distance_km, s=1, c='mediumspringgreen')
plt.plot(centroid_propagation_PD_filtered.Distance_km, res_PD.intercept + res_PD.slope*centroid_propagation_PD_filtered.Distance_km, 'blue', label=r'Fitted Line, Slope = {0}'.format(round(res_PD.slope,3)))
plt.plot(x, y, color='k',label = 'Line of Equality, Slope = 1.0')
plt.plot(averages['Pluvial-to-Drought']['Propagation'], averages['Pluvial-to-Drought']['Growth'], '*', color='blue', markersize=15, markeredgecolor='k')

plt.xticks(np.arange(0, 400, 100))
ax2.set_xticklabels(np.arange(0, 400, 100), fontsize=7)
plt.yticks(np.arange(0, 1100, 100))
ax2.set_yticklabels(np.arange(0, 1100, 100), fontsize=7)

ax2.set_xlabel('Centroid Propagation (km)', fontsize = 10)
ax2.set_ylabel('Event Growth (km)', fontsize = 10)	

plt.legend(loc='upper left', fontsize=7, framealpha=1) 	

plt.title("b) Pluvial-to-Drought Events", fontsize=10, loc ='left')

#Droughts
ax3 = fig.add_subplot(143)

plt.scatter(centroid_propagation_droughts_filtered.Distance_km, event_growth_droughts_filtered.Distance_km, s=1, c='mediumspringgreen')
plt.plot(centroid_propagation_droughts_filtered.Distance_km, res_droughts.intercept + res_droughts.slope*centroid_propagation_droughts_filtered.Distance_km, 'blue', label=r'Fitted Line, Slope = {0}'.format(round(res_droughts.slope,3)))
plt.plot(x, y, color='k',label = 'Line of Equality, Slope = 1.0')
plt.plot(averages['Drought']['Propagation'], averages['Drought']['Growth'], '*', color='blue', markersize=15, markeredgecolor='k')

plt.xticks(np.arange(0, 400, 100))
ax3.set_xticklabels(np.arange(0, 400, 100), fontsize=7)
plt.yticks(np.arange(0, 1100, 100))
ax3.set_yticklabels(np.arange(0, 1100, 100), fontsize=7)

ax3.set_xlabel('Centroid Propagation (km)', fontsize = 10)
ax3.set_ylabel('Event Growth (km)', fontsize = 10)	

plt.legend(loc='upper left', fontsize=7, framealpha=1)

plt.title("c) Drought Events", fontsize=10, loc='left')

#Pluvials
ax4 = fig.add_subplot(144)
plt.scatter(centroid_propagation_pluvials_filtered.Distance_km, event_growth_pluvials_filtered.Distance_km, s=1, c='mediumspringgreen')
plt.plot(centroid_propagation_pluvials_filtered.Distance_km, res_pluvials.intercept + res_pluvials.slope*centroid_propagation_pluvials_filtered.Distance_km, 'blue', label=r'Fitted Line, Slope = {0}'.format(round(res_pluvials.slope,3)))
plt.plot(x, y, color='k',label = 'Line of Equality, Slope = 1.0')
plt.plot(averages['Pluvial']['Propagation'], averages['Pluvial']['Growth'], '*', color='blue', markersize=15, markeredgecolor='k')

plt.xticks(np.arange(0, 400, 100))
ax4.set_xticklabels(np.arange(0, 400, 100), fontsize=7)
plt.yticks(np.arange(0, 1100, 100))
ax4.set_yticklabels(np.arange(0, 1100, 100), fontsize=7)

ax4.set_xlabel('Centroid Propagation (km)', fontsize = 10)
ax4.set_ylabel('Event Growth (km)', fontsize = 10)	

plt.legend(loc='upper left', fontsize=7, framealpha=1) 	

plt.title("d) Pluvial Events", fontsize=10, loc ='left')


plt.tight_layout()

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/event_propagation_vs_growth_cluster_length_normalized.eps', bbox_inches = 'tight', pad_inches = 0.1)    


