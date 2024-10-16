#########################################################################################
## File of functions needed and used for database creation
## Bryony Louise
## Last Edited: Wednesday October 16th 2024 
#########################################################################################
#Import Required Modules
#########################################################################################
import gzip
import numpy as np
#########################################################################################
#Functions
#########################################################################################
#save out files
def save(path, arr, **kwargs):
    with gzip.GzipFile(path, "w", **kwargs) as f:
        np.save(f, arr)
    return

#load in files
def load(path, **kwargs):
    with gzip.GzipFile(path, "r", **kwargs) as f:
        return np.load(f)
