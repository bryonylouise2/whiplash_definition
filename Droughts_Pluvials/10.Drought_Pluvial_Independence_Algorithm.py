#########################################################################################
## An objective post-processing algorithm to group "repeat" events,
## Bryony Louise Puxley
## Last Edited: Wednesday, August 13th, 2025 
## Input: Subsetted by area event file from 9. Database_Creation.py
## Output: A CSV file of either independent drought or independent pluvial events.
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
import pandas as pd
import scipy.stats as scs
import shapely.wkt
import os
#########################################################################################
#Import Functions
#########################################################################################
import functions

#########################################################################################
#Import Events - load previously made event files
#########################################################################################
events_droughts = pd.read_csv('/data2/bpuxley/droughts_and_pluvials/Events/events_droughts.csv')
events_pluvials = pd.read_csv('/data2/bpuxley/droughts_and_pluvials/Events/events_pluvials.csv')

#########################################################################################
#Call either Drought events or Pluvial Events
#########################################################################################
save_name = 'events_pluvials'
df = events_pluvials

#########################################################################################
#Independence Algorithm
#########################################################################################
## Group polygons into similar groups if they have overlapping dates, areas and spatial
## correlations of >= 0.5.

df['Event_No'] = np.nan #new column to store event number
df['Day_No'] = np.nan #new column to store day number within each event

events = pd.DataFrame(columns = df.columns)
event_no = 1

while True:
    num_events = len(df)
    while len(df) != 0:
        print(len(df))
        #first create matrix for correlation
        masked_events = functions.subset_events(df, 30)

        #calculate correlations and pull out groupedEventsc       
        coefs, iloc = functions.group_events(arr=masked_events, thresh=0.5)
        similar_events = df.iloc[iloc]
        similar_events.reset_index(drop=True, inplace=True) 
        similar_events.loc[:, 'Event_No'] = event_no
        event_no+=1
        
        subset_len = len(similar_events)
        
        for j in range(0,subset_len):
        	similar_events.loc[j, 'Day_No'] = j
        	
        similar_events.Event_No = similar_events.Event_No.astype(int) #convert to integer values
        similar_events.Day_No = similar_events.Day_No.astype(int) #convert to integer values

        events = pd.concat([events, similar_events], ignore_index=True)
        
        #remove events with correlation >= 0.5
        df.drop(iloc, inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    if len(df) == num_events:
        break
    
#########################################################################################
#Rearrange columns so 'Event_No' is column 1
#########################################################################################
cols = events.columns.tolist()
cols = cols[-2:] + cols[:-2]    
events = events[cols] 

#########################################################################################
#Save out file
#########################################################################################
events.to_csv(f'/data2/bpuxley/droughts_and_pluvials/Events/independent_%s.csv'%(save_name), index=False)
    
    
  
    
    
 
