#########################################################################################
## This script determines the appropriate number of clusters to group events into regions 
## across the CONUS (k-means clustering). This script uses the Elbow Method (Thorndike, 1953) 
## and the Silhouette Coefficient (Rousseeuw, 1987).
## Bryony Louise
## Last Edited: Monday, July 28th, 2025
## Input: Independent event files of Drought-to-Pluvial and Pluvial-to-Drought events.
## Output: PNG files. 1) The results of the Elbow Method, Silhouette Coefficient, Events
## retained and Hybrid Index to help determine the appropriate number of clusters. 2) The 
## average polygons of all the events in each cluster for the entire range of k-values to 
## view for spatial coherence, and 3) Supplementary Figure 4 from the journal article.
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
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import spei as si
import pandas as pd
import scipy.stats as scs
from shapely.ops import unary_union
import shapely.wkt
import os

from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

#########################################################################################
# Import Functions
#########################################################################################
import functions

#########################################################################################
# Import Events - load previously made event files
#########################################################################################
events_DP = pd.read_csv('/data2/bpuxley/Events/independent_events_DP.csv')
events_PD = pd.read_csv('/data2/bpuxley/Events/independent_events_PD.csv')

#Extract only Day 0 polygons
df_DP =  events_DP.iloc[np.where((events_DP.Day_No == 0))].reset_index(drop=True) 
df_PD =  events_PD.iloc[np.where((events_PD.Day_No == 0))].reset_index(drop=True) 

no_of_events_DP = np.nanmax(df_DP.Event_No)
no_of_events_PD = np.nanmax(df_PD.Event_No)

years = np.arange(1915,2021,1)

#########################################################################################
# Create a lat, lon grid
#########################################################################################
lat = np.arange(25.15625, 50.03125, 0.0625)
lon = np.arange(235.40625, 293.03125, 0.0625) 
lons, lats = np.meshgrid(lon,lat) #Livneh Grid

o,p = lats.shape

#########################################################################################
# Polygons
#########################################################################################
polygons_DP = [shapely.wkt.loads(i) for i in df_DP.geometry] #convert from nasty string of lat,lons to geometry object
polygons_PD = [shapely.wkt.loads(i) for i in df_PD.geometry] #convert from nasty string of lat,lons to geometry object

df_DP['polygon'] = polygons_DP
df_PD['polygon'] = polygons_PD

#Find the center of each polygon
df_DP['centroids'] = [poly.centroid for poly in polygons_DP] #get centroid of polygons
df_PD['centroids'] = [poly.centroid for poly in polygons_PD] #get centroid of polygons

#Find the boundary/exterior points of each polygon
df_DP['boundary_points'] = [list(poly.exterior.coords) for poly in polygons_DP]
df_PD['boundary_points'] = [list(poly.exterior.coords) for poly in polygons_PD]

#########################################################################################
# Create a vector consisting of 0s and 1s, where a grid point inside an event polygon is 
# assigned a 1, and outside an event polygon is assigned a 0.
#########################################################################################
def event_mask(lons, lats, polygon):
	array = np.ones((lats.shape[0], lons.shape[1])) #create an array of ones the size of the lat,lon grid
	mask = functions._mask_outside_region(lons, lats, polygon) #create polygon mask
	masked_array = np.ma.masked_array(array, ~mask)  #mask region outside polygon
	masked_array_filled = np.ma.filled(masked_array, 0) #fill mask with zeros
	
	return masked_array_filled
	
#Drought-to-Pluvial
event_vector_dp = np.zeros((no_of_events_DP, lats.shape[0], lons.shape[1]))

for i,(poly) in tqdm(enumerate(polygons_DP)):
	event_vector_dp[i] = event_mask(lons, lats, poly) #create the mask array for each polygon where 1 is inside polygon and 0 is outside polygon
	
		
#Pluvial-to-Drought
event_vector_pd = np.zeros((no_of_events_PD, lats.shape[0], lons.shape[1]))

for i,(poly) in tqdm(enumerate(polygons_PD)):
	event_vector_pd[i] = event_mask(lons, lats, poly) #create the mask array for each polygon where 1 is inside polygon and 0 is outside polygon

#########################################################################################
# K-means clustering
#########################################################################################
# Determine the Appropriate Number of Clusters
#########################################################################################
#########################################################################################
#Elbow Method (Thorndike 1953)
#########################################################################################
# This method looks at how the inertia (sum of squared distances from points to their cluster 
# center) decreases as you increase the number of clusters. The "elbow" point — where the 
# decrease slows — suggests a good number.
#########################################################################################
#Silhouette Coefficient (Rousseeuw 1987)
#########################################################################################
# This method measures how similar an event is to its own cluster compared to other clusters. 
# Higher is better (max is 1.0).

##########################################################################################
#Compute inertias, average silhouette, number of events clustered, and hybrid index
##########################################################################################
#set range of k values to test
k_range = np.arange(4,31,1)

# Flatten the spatial dimensions
data_reshaped_dp = event_vector_dp.reshape(no_of_events_DP, lats.shape[0] * lons.shape[1])  # Shape: (no_of_events, features)
data_reshaped_pd = event_vector_pd.reshape(no_of_events_PD, lats.shape[0] * lons.shape[1])  # Shape: (no_of_events, features)

def k_means_clustering(data, k_range):
	inertias = []
	silhouette_scores = []
	no_of_events_retained = []
	hybrid_index = []
	
	skipped_k = []  # Track k values that were skipped
	
	event_ids = np.arange(data.shape[0])
	cluster_matrix = pd.DataFrame(index=event_ids, columns=k_range)  # rows=events, cols=each k
	
	for k in tqdm(k_range):
		print(f"Processing k = {k}")
		
		# Reset the data for each k
		data_filtered = data.copy()
		original_indices = np.arange(data.shape[0])
		n = 1  # initialize to enter the loop
		
		while n != 0:
			kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
			kmeans.fit(data_filtered)
			
			## Compute individual silhouette scores
			labels = kmeans.predict(data_filtered)
			sil_scores = silhouette_samples(data_filtered, labels)
			print(np.nanmin(sil_scores))
			
			# Identify events with negative silhouette scores
			negative_mask = (sil_scores < 0)
			n = np.sum(negative_mask)
			print('n: '+str(n))
			
			if n != 0:
				data_filtered = data_filtered[~negative_mask]
				original_indices = original_indices[~negative_mask]
				print('len of data: '+str(len(data_filtered)))
				if len(data_filtered) <= k:
					print(f"Too few samples left for k = {k}. Skipping.")
					skipped_k.append(k)
					inertias.append(np.nan)
					silhouette_scores.append(np.nan)
					no_of_events_retained.append(np.nan)
					hybrid_index.append(np.nan)
					cluster_matrix[k] = np.nan  # Fill entire column with NaN
					break
			else:
				## Elbow Method ##
				inertias.append(kmeans.inertia_)
				
				## Silhouette Score ##
				score = silhouette_score(data_filtered, labels)
				silhouette_scores.append(score)
				
				## number of events clustered ##
				no_of_events_retained.append(len(labels))
				
				## Hybrid Index ##
				hybrid_index.append(len(labels) * score)
				
				# Fill in cluster assignments for the retained indices
				for idx, label in zip(original_indices, labels):
					cluster_matrix.at[idx, k] = label
	
	info_matrix = pd.DataFrame({'Number_of_Clusters': k_range, 'Inertia': inertias, 'Silhouette_Scores': silhouette_scores, 'No_of_Events_Retained':no_of_events_retained, 'Hybrid_Index':hybrid_index})
	cluster_matrix.fillna(np.nan, inplace=True)  # or .fillna(-1)
	
	return info_matrix, cluster_matrix, skipped_k
            	
info_matrix_dp, cluster_matrix_dp, skipped_k_dp = k_means_clustering(data_reshaped_dp, k_range)
cluster_matrix_dp.to_csv(f'/data2/bpuxley/Events/cluster_matrix_DP.csv', index=False)
info_matrix_dp.to_csv(f'/data2/bpuxley/Events/info_matrix_DP.csv', index=False)

info_matrix_pd, cluster_matrix_pd,skipped_k_pd = k_means_clustering(data_reshaped_pd, k_range)
cluster_matrix_pd.to_csv(f'/data2/bpuxley/Events/cluster_matrix_PD.csv', index=False)
info_matrix_pd.to_csv(f'/data2/bpuxley/Events/info_matrix_PD.csv', index=False)

##########################################################################################
#Read in Data
##########################################################################################
cluster_matrix_dp = pd.read_csv('/data2/bpuxley/Events/cluster_matrix_DP.csv')
cluster_matrix_pd = pd.read_csv('/data2/bpuxley/Events/cluster_matrix_PD.csv')

info_matrix_dp = pd.read_csv('/data2/bpuxley/Events/info_matrix_DP.csv')
info_matrix_pd = pd.read_csv('/data2/bpuxley/Events/info_matrix_PD.csv')

##########################################################################################
#Average the polygons of each cluster
##########################################################################################
new_df_dp = df_DP[['Event_No','geometry','polygon','centroids','boundary_points']]
new_df_dp = pd.concat([new_df_dp, cluster_matrix_dp], axis=1)

new_df_pd = df_PD[['Event_No','geometry','polygon','centroids','boundary_points']]
new_df_pd = pd.concat([new_df_pd, cluster_matrix_pd], axis=1)

def find_avg_cluster_polygons(df, k_range, cluster_matrix):
	cluster_points = pd.concat([pd.DataFrame(columns=['cluster_no']), pd.DataFrame(columns=cluster_matrix.columns)])
	cluster_points.cluster_no = np.arange(0,30,1)
	
	cluster_polygons = cluster_points.copy()
	cluster_areas = cluster_points.copy()

	for k in tqdm(k_range):
		print(f"Processing k = {k}")
		column_of_interest = df[str(k)]
	
		for i in range(0, k):
			subset_ind = np.where((column_of_interest == i))[0]
			subset =  df.iloc[subset_ind]
			
			if len(subset) > 0:
				polygons = subset.polygon
			
				cluster_shape = np.zeros((lats.shape[0], lons.shape[1]))

				for j,(poly) in enumerate(polygons):
					masked_array = event_mask(lons, lats, poly) #create the mask array for each polygon where 1 is inside polygon and 0 is outside polygon
					cluster_shape = np.ma.add(cluster_shape, masked_array) #add together each masked array
				
				a, p = functions.calc_area(lon, lat, cluster_shape, isopleth=1, area_threshold=0)
				
				#Assign all to relevant dataframes
				cluster_points.loc[i, str(k)] = [cluster_shape]
				cluster_polygons.loc[i, str(k)] = [[p]]
				#cluster_areas.loc[i, str(k)] = [[a]]
				
			else:
				cluster_points.loc[i, str(k)] = np.nan
				cluster_polygons.loc[i, str(k)] = np.nan
				#cluster_areas.loc[i, str(k)] = np.nan
	
	return cluster_points, cluster_polygons, cluster_areas

def get_contour(grid_lon, grid_lat, density, isopleth, **kwargs):
    ax = plt.axes(projection=TARGET_PROJ, **kwargs)
    im = ax.contour(grid_lon, grid_lat, density, levels=[isopleth], transform=ORIG_PROJ)
    return im
	
cluster_points_dp, cluster_polygons_dp, cluster_areas_dp = find_avg_cluster_polygons(new_df_dp, k_range, cluster_matrix_dp)
cluster_points_pd, cluster_polygons_pd, cluster_areas_pd = find_avg_cluster_polygons(new_df_pd, k_range, cluster_matrix_pd)

cluster_points_dp.to_csv(f'/data2/bpuxley/Events/cluster_points_DP.csv', index=False)
cluster_polygons_dp.to_csv(f'/data2/bpuxley/Events/cluster_polygons_DP.csv', index=False)
cluster_areas_dp.to_csv(f'/data2/bpuxley/Events/cluster_areas_DP.csv', index=False)

cluster_points_pd.to_csv(f'/data2/bpuxley/Events/cluster_points_PD.csv', index=False)
cluster_polygons_pd.to_csv(f'/data2/bpuxley/Events/cluster_polygons_PD.csv', index=False)
cluster_areas_pd.to_csv(f'/data2/bpuxley/Events/cluster_areas_PD.csv', index=False)


##########################################################################################
#Read in Data
##########################################################################################
cluster_polygons_dp = pd.read_csv('/data2/bpuxley/Events/cluster_polygons_DP.csv')
cluster_polygons_pd = pd.read_csv('/data2/bpuxley/Events/cluster_polygons_PD.csv')

#########################################################################################
#Plot
#########################################################################################
#########################################################################################
#a) Elbow Method, b) Silhouette Score, c) Number of Events Retained, and d) HI Index
#########################################################################################
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

## Elbow Method ##
ax1 = fig.add_subplot(221)
plt.plot(k_range, info_matrix_dp['Inertia'], color='k', marker='o', markersize=5, label = 'drought-to-pluvial')
plt.plot(k_range, info_matrix_pd['Inertia'], color='purple', marker='o', markersize=5, label = 'pluvial-to-drought')

plt.grid(True)

plt.xticks(np.arange(4, 31, 2))
plt.yticks(np.arange(0, 6000000, 500000))

plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

plt.legend(loc='upper right', fontsize=7, framealpha=1)

plt.title('a) Sum of Squared Distances \nby $\\it{k}$ Clusters', loc = "left",fontsize=10)

## Silhouette Score ##
ax2 = fig.add_subplot(222)
plt.plot(k_range, info_matrix_dp['Silhouette_Scores'], color='k', marker='o', markersize=5, label = 'drought-to-pluvial')
plt.plot(k_range, info_matrix_pd['Silhouette_Scores'], color='purple', marker='o', markersize=5, label = 'pluvial-to-drought')

plt.grid(True)

plt.xticks(np.arange(4, 31, 2))
plt.yticks(np.arange(0.05, 0.45, 0.05))

plt.xlabel('Number of clusters')
plt.ylabel('Average Silhouette')

plt.legend(loc='upper right', fontsize=7, framealpha=1)

plt.title('b) Silhouette Score \nby $\\it{k}$ Clusters', loc = "left",fontsize=10)

## number of events retained ##
ax3 = fig.add_subplot(223)
plt.plot(k_range, info_matrix_dp['No_of_Events_Retained'], color='k', marker='o', markersize=5, label = 'drought-to-pluvial')
plt.plot(k_range, info_matrix_pd['No_of_Events_Retained'], color='purple', marker='o', markersize=5, label = 'pluvial-to-drought')

plt.grid(True)

plt.xticks(np.arange(4, 31, 2))
plt.yticks(np.arange(0, 1200, 100))

plt.xlabel('Number of clusters')
plt.ylabel('Number of Events Retained')

plt.legend(loc='upper right', fontsize=7, framealpha=1)

plt.title('c) Number of Events Retained \nby $\\it{k}$ Clusters', loc = "left",fontsize=10)

## h-index ##
ax4 = fig.add_subplot(224)
plt.plot(k_range, info_matrix_dp['Hybrid_Index'], color='k', marker='o', markersize=5, label = 'drought-to-pluvial')
plt.plot(k_range, info_matrix_pd['Hybrid_Index'], color='purple', marker='o', markersize=5, label = 'pluvial-to-drought')

plt.grid(True)

plt.xticks(np.arange(4, 31, 2))
plt.yticks(np.arange(0, 120, 10))

plt.xlabel('Number of clusters')
plt.ylabel('Hybrid Index (HI)')

plt.legend(loc='upper right', fontsize=7, framealpha=1)

plt.title('d) Product of Events and Silhouette \nby $\\it{k}$ Clusters', loc = "left",fontsize=10)

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/k_means_combined.eps', bbox_inches = 'tight', pad_inches = 0.1)    

#########################################################################################
#Average polygons for k = X clusters across the CONUS
#########################################################################################
n_colors = 30
cmap = cm.get_cmap('tab20', n_colors)  # Discretize it
colors = [cmap(i) for i in range(n_colors)]  # List of RGBA tuples

colors = ['purple', 'blue', 'green', 'orange', 'black', 'red', 'saddlebrown', 'gold', 'pink', 'deeppink', 
			'deepskyblue', 'springgreen', 'olive', 'tan', 'grey', 'darkred', 'cyan', 'mediumpurple', 'tomato', 'chocolate',
			'yellow', 'lawngreen', 'lavender', 'plum',' fuchsia', 'palevioletred', 'rosybrown',' darkcyan', 'aquamarine', 'navy']

for k in tqdm(k_range):
	print(f"Processing k = {k}")

	fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)
	
	#Drought-to-Pluvial
	ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

	ax1.add_feature(cfeature.COASTLINE)
	ax1.add_feature(cfeature.BORDERS, linewidth=1)
	ax1.add_feature(cfeature.STATES, edgecolor='black')

	ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())
	
	for i in range(0, k):
		print(i)
		color = colors[i]
		print(color)
		polygons = cluster_polygons_dp[str(k)][i][0][0]
		plot_polygons(ax1, polygons, color, linewidth=2)
			
	plt.title("Event Clusters ($\\it{k}$ = %s)"%(k), loc = "left", fontsize=10)
	plt.title("Drought-to-Pluvial", loc = 'right',fontsize=10)

	#Pluvial-to-Drought
	ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

	ax2.add_feature(cfeature.COASTLINE)
	ax2.add_feature(cfeature.BORDERS, linewidth=1)
	ax2.add_feature(cfeature.STATES, edgecolor='black')

	ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())
	
	for i in range(0, k):
		print(i)
		color = colors[i]
		print(color)
		polygons = cluster_polygons_pd[str(k)][i][0][0]
		plot_polygons(ax2, polygons, color, linewidth=2)

	plt.title("Event Clusters ($\\it{k}$ = %s)"%(k), loc = "left", fontsize=10)
	plt.title("Pluvial-to-Drought", loc = 'right',fontsize=10)

	plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/Clusters/k_means_event_clusters_%s.png'%(k), bbox_inches = 'tight', pad_inches = 0.1)    


def plot_polygons(ax, polygons, colors, **kwargs):
	for i, polygon in enumerate(polygons):
		#color = colors[i % len(colors)]  # Wrap around if needed
		print(color)
		if polygon.geom_type == 'MultiPolygon':
			geoms = polygon.geoms
		else:
			geoms = [polygon]
		for poly in geoms:
			x, y = poly.exterior.xy
			ax.plot(x, y, transform=ccrs.PlateCarree(), color=color, **kwargs)

def is_valid_geom(g):
    return g is not None and g == g and isinstance(g, (Polygon, MultiPolygon))

#########################################################################################
#Supplementary Figure (k=7 and k=6)
#########################################################################################
n_colors = 30
cmap = cm.get_cmap('tab20', n_colors)  # Discretize it
colors = [cmap(i) for i in range(n_colors)]  # List of RGBA tuples

colors = ['purple', 'blue', 'green', 'orange', 'black', 'red', 'saddlebrown', 'gold', 'pink', 'deeppink', 
			'deepskyblue', 'springgreen', 'olive', 'tan', 'grey', 'darkred', 'cyan', 'mediumpurple', 'tomato', 'chocolate',
			'yellow', 'lawngreen', 'lavender', 'plum',' fuchsia', 'palevioletred', 'rosybrown',' darkcyan', 'aquamarine', 'navy']

fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

k=7
print(f"Processing k = {k}")

#Drought-to-Pluvial
ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())
		
for i in range(0, k):
	print(i)
	color = colors[i]
	print(color)
	polygons = cluster_polygons_dp[str(k)][i][0][0]
	plot_polygons(ax1, polygons, color, linewidth=2)
			
plt.title("a) Event Clusters ($\\it{k}$ = %s)"%(k), loc = "left", fontsize=10)
plt.title("Drought-to-Pluvial", loc = 'right',fontsize=10)

#Pluvial-to-Drought
ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linewidth=1)
ax2.add_feature(cfeature.STATES, edgecolor='black')

ax2.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())
		
for i in range(0, k):
	print(i)
	color = colors[i]
	print(color)
	polygons = cluster_polygons_pd[str(k)][i][0][0]
	plot_polygons(ax2, polygons, color, linewidth=2)

plt.title("b) Event Clusters ($\\it{k}$ = %s)"%(k), loc = "left", fontsize=10)
plt.title("Pluvial-to-Drought", loc = 'right',fontsize=10)

k=6
print(f"Processing k = {k}")

#Drought-to-Pluvial
ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax3.add_feature(cfeature.COASTLINE)
ax3.add_feature(cfeature.BORDERS, linewidth=1)
ax3.add_feature(cfeature.STATES, edgecolor='black')

ax3.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())
		
for i in range(0, k):
	print(i)
	color = colors[i]
	print(color)
	polygons = cluster_polygons_dp[str(k)][i][0][0]
	plot_polygons(ax3, polygons, color, linewidth=2)
			
plt.title("c) Event Clusters ($\\it{k}$ = %s)"%(k), loc = "left", fontsize=10)
plt.title("Drought-to-Pluvial", loc = 'right',fontsize=10)

#Pluvial-to-Drought
ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax4.add_feature(cfeature.COASTLINE)
ax4.add_feature(cfeature.BORDERS, linewidth=1)
ax4.add_feature(cfeature.STATES, edgecolor='black')

ax4.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())
		
for i in range(0, k):
	print(i)
	color = colors[i]
	print(color)
	polygons = cluster_polygons_pd[str(k)][i][0][0]
	plot_polygons(ax4, polygons, color, linewidth=2)

plt.title("d) Event Clusters ($\\it{k}$ = %s)"%(k), loc = "left", fontsize=10)
plt.title("Pluvial-to-Drought", loc = 'right',fontsize=10)

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/Clusters/k_means_event_clusters_supplementary.png', bbox_inches = 'tight', pad_inches = 0.1)    


'''
Extra - Drought-to-Pluvial and Pluvial-to-Drought Events Seperately
#########################################################################################
#Drought-to-Pluvial (Summary)
#########################################################################################
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

## Elbow Method ##
ax1 = fig.add_subplot(221)
plt.plot(k_range, info_matrix_dp['Inertia'], color='k', marker='o')

plt.grid(True)

plt.xticks(np.arange(4, 31, 2))
plt.yticks(np.arange(0, 6000000, 500000))

plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

plt.title('a) Sum of Squared Distances \nby $\\it{k}$ Clusters', loc = "left",fontsize=10)
plt.title("Drought-to-Pluvial", loc = 'right',fontsize=10)


## Silhouette Score ##
ax2 = fig.add_subplot(222)
plt.plot(k_range, info_matrix_dp['Silhouette_Scores'], color='k', marker='o')

plt.grid(True)

plt.xticks(np.arange(4, 31, 2))
plt.yticks(np.arange(0.05, 0.35, 0.05))

plt.xlabel('Number of clusters')
plt.ylabel('Average Silhouette')

plt.title('b) Silhouette Score \nby $\\it{k}$ Clusters', loc = "left",fontsize=10)
plt.title("Drought-to-Pluvial", loc = 'right',fontsize=10)

## number of events retained ##
ax3 = fig.add_subplot(223)
plt.plot(k_range, info_matrix_dp['No_of_Events_Retained'], color='k', marker='o')

plt.grid(True)

plt.xticks(np.arange(4, 31, 2))
plt.yticks(np.arange(0, 1200, 100))

plt.xlabel('Number of clusters')
plt.ylabel('Number of Events Retained')

plt.title('c) Number of Events Retained \nby $\\it{k}$ Clusters', loc = "left",fontsize=10)
plt.title("Drought-to-Pluvial", loc = 'right',fontsize=10)

## h-index ##
ax4 = fig.add_subplot(224)
plt.plot(k_range, info_matrix_dp['Hybrid_Index'], color='k', marker='o')

plt.grid(True)

plt.xticks(np.arange(4, 31, 2))
plt.yticks(np.arange(0, 120, 10))

plt.xlabel('Number of clusters')
plt.ylabel('Hybrid Index (HI)')

plt.title('d) Product of Events and Silhouette \nby $\\it{k}$ Clusters', loc = "left",fontsize=10)
plt.title("Drought-to-Pluvial", loc = 'right',fontsize=10)

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/k_means_drought_to_pluvial.png', bbox_inches = 'tight', pad_inches = 0.1)    

#########################################################################################
#Pluvial-to-Drought (Summary)
#########################################################################################
fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)

## Elbow Method ##
ax1 = fig.add_subplot(221)
plt.plot(k_range, info_matrix_pd['Inertia'], color='k', marker='o')

plt.grid(True)

plt.xticks(np.arange(4, 31, 2))
plt.yticks(np.arange(0, 6000000, 500000))

plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

plt.title('a) Sum of Squared Distances \nby $\\it{k}$ Clusters', loc = "left",fontsize=10)
plt.title("Pluvial-to-Drought", loc = 'right',fontsize=10)

## Silhouette Score ##
ax2 = fig.add_subplot(222)
plt.plot(k_range, info_matrix_pd['Silhouette_Scores'], color='k', marker='o')

plt.grid(True)

plt.xticks(np.arange(4, 31, 2))
plt.yticks(np.arange(0.05, 0.35, 0.05))

plt.xlabel('Number of clusters')
plt.ylabel('Average Silhouette')

plt.title('b) Silhouette Score \nby $\\it{k}$ Clusters', loc = "left",fontsize=10)
plt.title("Pluvial-to-Drought", loc = 'right',fontsize=10)


## number of events retained ##
ax3 = fig.add_subplot(223)
plt.plot(k_range, info_matrix_pd['No_of_Events_Retained'], color='k', marker='o')

plt.grid(True)

plt.xticks(np.arange(4, 31, 2))
plt.yticks(np.arange(0, 1200, 100))

plt.xlabel('Number of clusters')
plt.ylabel('Number of Events Retained')

plt.title('c) Number of Events Retained \nby $\\it{k}$ Clusters', loc = "left",fontsize=10)
plt.title("Pluvial-to-Drought", loc = 'right',fontsize=10)


## h-index ##
ax4 = fig.add_subplot(224)
plt.plot(k_range, info_matrix_pd['Hybrid_Index'], color='k', marker='o')

plt.grid(True)

plt.xticks(np.arange(4, 31, 2))
plt.yticks(np.arange(0, 120, 10))

plt.xlabel('Number of clusters')
plt.ylabel('Hybrid Index (HI)')

plt.title('d) Product of Events and Silhouette\nby $\\it{k}$ Clusters', loc = "left",fontsize=10)
plt.title("Pluvial-to-Drought", loc = 'right',fontsize=10)

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/k_means_pluvial_to_drought.png', bbox_inches = 'tight', pad_inches = 0.1)    
