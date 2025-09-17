#########################################################################################
## A script to cluster drought and pluvial events only into regions across the CONUS 
## (k-means clustering) based on the appropriate number of clusters determined in 
## 11.Determine_Clusters.py. The script uses the largest areal overlap (intersection area) 
## to assign events to clusters.
## Bryony Louise Puxley
## Last Edited: Monday, July 28th, 2025
## Input: Independent event files of Drought and Pluvial events and chosen Cluster Number.
## Output: Updated independent event files (drought and pluvial only) with a column for 
## cluster number assignment.  
#########################################################################################
# Import Required Modules
#########################################################################################
import os
import xesmf as xe
import numpy as np
import pandas as pd
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

import scipy.stats as scs
from shapely.ops import unary_union
import shapely.wkt

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
events_droughts = pd.read_csv('/data2/bpuxley/droughts_and_pluvials/Events/independent_events_droughts.csv')
events_pluvials = pd.read_csv('/data2/bpuxley/droughts_and_pluvials/Events/independent_events_pluvials.csv')

#Extract only Day 0 polygons
df_droughts =  events_droughts.iloc[np.where((events_droughts.Day_No == 0))].reset_index(drop=True) 
df_pluvials =  events_pluvials.iloc[np.where((events_pluvials.Day_No == 0))].reset_index(drop=True) 

no_of_events_droughts = np.nanmax(df_droughts.Event_No)
no_of_events_pluvials = np.nanmax(df_pluvials.Event_No)

years = np.arange(1915,2021,1)

#########################################################################################
# Import Cluster Data
#########################################################################################
cluster_matrix_dp = pd.read_csv('/data2/bpuxley/Events/cluster_matrix_DP.csv')

cluster_no = 7 #choose cluster number
cluster_column_dp = cluster_matrix_dp[str(cluster_no)]

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
polygons_droughts = [shapely.wkt.loads(i) for i in df_droughts.geometry] #convert from nasty string of lat,lons to geometry object
polygons_pluvials = [shapely.wkt.loads(i) for i in df_pluvials.geometry] #convert from nasty string of lat,lons to geometry object

df_droughts['polygon'] = polygons_droughts
df_pluvials['polygon'] = polygons_pluvials

#Find the center of each polygon
df_droughts['centroids'] = [poly.centroid for poly in polygons_droughts] #get centroid of polygons
df_pluvials['centroids'] = [poly.centroid for poly in polygons_pluvials] #get centroid of polygons

#Find the boundary/exterior points of each polygon
df_droughts['boundary_points'] = [list(poly.exterior.coords) for poly in polygons_droughts]
df_pluvials['boundary_points'] = [list(poly.exterior.coords) for poly in polygons_pluvials]

#########################################################################################
#Read in Cluster Data
#########################################################################################
cluster_polys = pd.read_csv('/data2/bpuxley/Events/cluster_polygons_final.csv')

cluster_polys['avg_poly'] = cluster_polys['avg_poly'].apply(shapely.wkt.loads)

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

df_droughts = assign_cluster_no(df_droughts, cluster_polygons)
df_pluvials = assign_cluster_no(df_pluvials, cluster_polygons)

###########################################################################################
# Duplicate Cluster Numbers to match the length of each event and match the events database
###########################################################################################

def duplicate_cluster_numbers(initial_df, final_df, no_of_events):
	event_cluster_nums = []
	
	for i in tqdm(range(0, no_of_events)):
		subset_ind = np.where((final_df.Event_No == i+1))[0]
		subset = final_df.iloc[subset_ind]
	
		event_length = len(subset)

		cluster_no = initial_df['cluster_no'][i]

		event_cluster_nums.extend([cluster_no]*event_length)
		
	final_df['cluster_no'] = event_cluster_nums
	
	return final_df

events_droughts = duplicate_cluster_numbers(df_droughts, events_droughts, no_of_events_droughts)
events_pluvials = duplicate_cluster_numbers(df_pluvials, events_pluvials, no_of_events_pluvials)

events_droughts.to_csv(f'/data2/bpuxley/droughts_and_pluvials/Events/independent_events_droughts.csv', index=False)
events_pluvials.to_csv(f'/data2/bpuxley/droughts_and_pluvials/Events/independent_events_pluvials.csv', index=False)


