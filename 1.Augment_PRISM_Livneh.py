#########################################################################################
## Augment PRISM and Livneh Datasets  
## Bryony Louise
## Last Edited: Wednesday May 29th 2024 
#########################################################################################
#Required Modules for this Function
#########################################################################################
import xesmf as xe
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime, timedelta, date
from netCDF4 import Dataset, num2date, MFDataset

##########################################################################################
#Bilinearly interpolate PRISM precipitation data from a native 4-km grid onto Livneh's grid 
#of about 6 km using Python library xESMF (Zhuang et al. 2020). 
##########################################################################################
def interp(data):
	'''Bilinearly interpolate PRISM grid to Livneh grid.
	
	Parameters
	----------
	data : array
		Array to be interpolated (PRISM)
		
	split : boolean
		If False (default). the time period does not begin in 2011 and end in 2012 (crossover from Livneh to PRISM)

	firstPiece : NoneType or array
        Ignore if split is False. If split is True, an array should be given holding Livneh data.

	'''
	
	prism_pathfile = "/data/deluge/reanalysis/REANALYSIS/PRISM/daily/prec.2020.nc" #gridIn
	livneh_pathfile = "/data/deluge/reanalysis/REANALYSIS/Livneh/prec.2011.nc"  #gridOut

	with Dataset(prism_pathfile, 'r') as nc:
		prismLat = nc.variables['lat'][:]
		prismLon = nc.variables['lon'][:]
		
	with Dataset(livneh_pathfile, 'r') as nc:
		livnehLat = nc.variables['lat'][:]
		livnehLon = nc.variables['lon'][:]
		
	gridIn = {'lat':prismLat, 'lon':prismLon+360}
	gridOut = {'lat':livnehLat, 'lon':livnehLon}
	regridder = xe.Regridder(gridIn, gridOut, method='bilinear')
	prismRegrid = regridder(data)

	return prismRegrid
