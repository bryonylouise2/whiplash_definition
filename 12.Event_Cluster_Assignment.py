#########################################################################################
## A script to cluster events into regions across the CONUS (k-means clustering) based on
## the appropriate number of clusters determined in 11.Determine_Clusters.py.
## Bryony Louise Puxley
## Last Edited: Monday, July 28th, 2025
## Input: Independent event files of Drought-to-Pluvial and Pluvial-to-Drought events and 
## chosen Cluster Number.
## Output: 
#########################################################################################
# Import Required Modules
#########################################################################################
import xesmf as xe
import numpy as np
import xarray as xr
from tqdm import tqdm
import time
from datetime import datetime, timedelta, date
from netCDF4 import Dataset, num2date, MFDataset
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
    
def find_avg_cluster_polygons(df, k, cluster_matrix):
	cluster_points = pd.concat([pd.DataFrame(columns=['cluster_no']), pd.DataFrame(columns=[str(k)])])
	cluster_points.cluster_no = np.arange(0,k,1)
	
	cluster_polygons = cluster_points.copy()
	cluster_areas = cluster_points.copy()

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

def event_mask(lons, lats, polygon):
	array = np.ones((lats.shape[0], lons.shape[1])) #create an array of ones the size of the lat,lon grid
	mask = functions._mask_outside_region(lons, lats, polygon) #create polygon mask
	masked_array = np.ma.masked_array(array, ~mask)  #mask region outside polygon
	masked_array_filled = np.ma.filled(masked_array, 0) #fill mask with zeros
	
	return masked_array_filled

# Function to assign best-fit cluster by largest intersection area
def assign_cluster(event_poly, clusters_df):
    best_cluster_id = None
    max_area = 0
    for _, row in clusters_df.iterrows():
        inter = event_poly.intersection(row['avg_poly'])
        if inter.area > max_area:
            max_area = inter.area
            best_cluster_id = row['cluster_no']
    return best_cluster_id


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
# Import Cluster Data
#########################################################################################
cluster_matrix_dp = pd.read_csv('/data2/bpuxley/Events/cluster_matrix_DP.csv')

cluster_no = 7 #choose cluster number
cluster_column_dp = cluster_matrix_dp[str(cluster_no)]

df_DP['cluster_no_org'] = cluster_column_dp

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
# Find Cluster Polygons
#########################################################################################
new_df_dp = df_DP[['Event_No', 'geometry', 'polygon', 'centroids', 'boundary_points']]
new_df_dp = pd.concat([new_df_dp, cluster_matrix_dp], axis=1)

cluster_no = 7 #choose cluster number

cluster_points_dp, cluster_polygons_dp, cluster_areas_dp = find_avg_cluster_polygons(new_df_dp, cluster_no, cluster_matrix_dp)

#########################################################################################
# Change Cluster Numbers to something more intuitive
#########################################################################################
cluster_polygons_dp.loc[5, 'cluster_no'] = 1
cluster_polygons_dp.loc[2, 'cluster_no'] = 2
cluster_polygons_dp.loc[0, 'cluster_no'] = 3
cluster_polygons_dp.loc[6, 'cluster_no'] = 4
cluster_polygons_dp.loc[1, 'cluster_no'] = 5
cluster_polygons_dp.loc[4, 'cluster_no'] = 6
cluster_polygons_dp.loc[3, 'cluster_no'] = 7

cluster_polygons_dp = cluster_polygons_dp.sort_values(by='cluster_no').reset_index(drop=True)

largest_poly = []

for i in range(0, cluster_no):
	print(i)
	polygons = cluster_polygons_dp[str(cluster_no)][i][0][0]
	all_polys = []
	for polygon in polygons:
		if polygon.geom_type == 'MultiPolygon':
			all_polys.extend(polygon.geoms)
		else:
			all_polys.append(polygon)
			
	# Find the largest polygon by area
	largest_poly.append(max(all_polys, key=lambda p: p.area))

cluster_polygons_dp['avg_poly'] = largest_poly

# Convert polygons to WKT
cluster_polygons_dp['avg_poly'] = cluster_polygons_dp['avg_poly'].apply(lambda geom: geom.wkt)

cluster_polygons_dp.to_csv(f'/data2/bpuxley/Events/cluster_polygons_final.csv', index=False)

#########################################################################################
#Read in Cluster Data
#########################################################################################
cluster_polys = pd.read_csv('/data2/bpuxley/Events/cluster_polygons_final.csv')

cluster_polys['avg_poly'] = cluster_polys['avg_poly'].apply(shapely.wkt.loads)

#########################################################################################
#Plot Cluster Plot
#########################################################################################
colors = ['red', 'green', 'purple', 'saddlebrown', 'blue', 'black', 'orange', 'gold', 'pink', 'deeppink', 
			'deepskyblue', 'springgreen', 'olive', 'tan', 'grey', 'darkred', 'cyan', 'mediumpurple','tomato','chocolate',
			'yellow','lawngreen','lavender','plum','fuchsia','palevioletred','rosybrown','darkcyan','aquamarine','navy']


print(f"Processing k = {cluster_no}")

fig = plt.figure(figsize = (10,6), dpi = 300, tight_layout =True)
	
ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree()) #ccrs.LambertConformal())

ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linewidth=1)
ax1.add_feature(cfeature.STATES, edgecolor='black')

ax1.set_extent([-130, -60, 21, 50], crs=ccrs.PlateCarree())
		
for i in range(0, cluster_no):
	print(i)
	color = colors[i]
	print(color)
	polygons = cluster_polys['avg_poly'][i]
	x, y = polygons.exterior.xy
	ax1.plot(x, y, transform=ccrs.PlateCarree(), color=color)
	
	# Get centroid and add label
	centroid = polygons.centroid
	cx, cy = centroid.x, centroid.y
	
	ax1.text(cx, cy, str(i+1), transform=ccrs.PlateCarree(),
			ha='center', va='center', fontsize=10,
			fontweight='bold', color=color,
			bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1', alpha=1))		

			
plt.title("Event Clusters ($\\it{k}$ = %s)"%(cluster_no), loc = "left", fontsize=10)

plt.savefig('/home/bpuxley/Definition_and_Climatology/Plots/Clusters/k_means_event_clusters.eps', bbox_inches = 'tight', pad_inches = 0.1)    

#########################################################################################
# Assign Events to the 7 clusters - largest areal overlap (intersection area)
# event polygon overlaps with cluster polygon, and you assign it to the one with the largest intersecting area.
#########################################################################################
def assign_cluster_no(df, cluster_polygons):
	#Check for invalid events
	invalid_events = df[~df['polygon'].apply(lambda g: g.is_valid)]
	print('Invalid Events:')
	print(invalid_events)
	
	# Fix all event polygons
	df['polygon'] = df['polygon'].apply(lambda g: g.buffer(0) if not g.is_valid else g)
	
	# Apply to event polygons
	df['cluster_no'] = df['polygon'].apply(lambda poly: assign_cluster(poly, cluster_polygons))

	return df

###########################################################################################################
cluster_polygons = cluster_polys.drop('7', axis=1)

df_DP = assign_cluster_no(df_DP, cluster_polygons)
df_PD = assign_cluster_no(df_PD, cluster_polygons)

###########################################################################################
# Duplicate Cluster Numbers to match the length of each event and match the events database
###########################################################################################
def duplicate_cluster_numbers(initial_df, final_df, no_of_events):
	event_cluster_nums = []
	
	for i in tqdm(range(0, no_of_events)):
		subset_ind = np.where((final_df.Event_No == i+1))[0]
		subset = df.iloc[subset_ind]
	
		event_length = len(subset)

		cluster_no = initial_df['cluster_no'][i]

		event_cluster_nums.extend([cluster_no]*event_length)
		
	final_df['cluster_no'] = event_cluster_nums
	
	return final_df

events_DP = duplicate_cluster_numbers(df_DP, events_DP, no_of_events_DP)
events_PD = duplicate_cluster_numbers(df_PD, events_PD, no_of_events_PD)

events_DP.to_csv(f'/data2/bpuxley/Events/independent_events_DP.csv', index=False)
events_PD.to_csv(f'/data2/bpuxley/Events/independent_events_PD.csv', index=False)

