#########################################################################################
## File of functions needed and used for database creation
## Bryony Louise
## Last Edited: Friday February 28th 2025
#########################################################################################
#Import Required Modules
#########################################################################################
import gzip
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from shapely.geometry import Polygon
import shapely.vectorized
from tqdm import tqdm
import scipy.stats as scs
import statsmodels.api as sm
import xarray as xr
from datetime import datetime, timedelta, date

import matplotlib.pyplot as plt
from matplotlib.path import Path
import cartopy.crs as ccrs


# GLOBALS
ORIG_PROJ = ccrs.PlateCarree()
TARGET_PROJ = ccrs.AlbersEqualArea(central_latitude=37.5, central_longitude=265)


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

''' -> will likely need for climate model things
def _determine_bandwidth(lon, lat):
    x_spacing = np.diff(lon)[0]
    y_spacing = np.diff(lat)[0]
    spacing = max(x_spacing, y_spacing)
    if spacing < 0.5:
        return 0.02
    elif (spacing >= 0.5) and (spacing < 0.75):
        return 0.025
    elif (spacing >= 0.75) and (spacing < 1.25):
        return 0.03
    elif (spacing >= 1.25) and (spacing < 1.75):
        return 0.035
    elif (spacing >= 1.75) and (spacing < 2.25):
        return 0.04
    elif (spacing >= 2.25) and (spacing < 2.75):
        return 0.045
    elif (spacing >= 2.75) and (spacing < 3.25):
        return 0.05
    else:
        return 0.06
'''

def kde(orig_lon, orig_lat, grid_lon, grid_lat, extreme, **kwargs):
    kwargs.setdefault("bandwidth", 0.02)#_determine_bandwidth(orig_lon, orig_lat))
    kwargs.setdefault("metric", "haversine") #formula to calculate the distance - use as we as on a sphere
    kwargs.setdefault("kernel", "epanechnikov")
    kwargs.setdefault("algorithm", "ball_tree")

    # get locations of extreme points and convert to radians - we are on sohere and want to use Haversine formula
    orig_lon, orig_lat = np.meshgrid(orig_lon, orig_lat)
    ex_locs = np.where(extreme == 1)
    ex_points = np.zeros((ex_locs[0].size, 2))
    ex_points[:, 0] = orig_lat[ex_locs]
    ex_points[:, 1] = orig_lon[ex_locs]
    ex_points *= np.pi / 180.0

    # setup grid to put kde density onto and convert to radians
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    kde_grid = np.zeros((grid_lon.size, 2))
    kde_grid[:, 0] = grid_lat.flatten()
    kde_grid[:, 1] = grid_lon.flatten()
    kde_grid *= np.pi / 180.0

    KDE = KernelDensity(**kwargs)
    KDE.fit(ex_points)
    density = np.exp(KDE.score_samples(kde_grid))
    # divide by max to normalize density field
    density = density / np.nanmax(density)
    density = density.reshape(grid_lon.shape)
    return density

def get_contour(grid_lon, grid_lat, density, isopleth, **kwargs):
    ax = plt.axes(projection=TARGET_PROJ, **kwargs)
    im = ax.contour(grid_lon, grid_lat, density, levels=[isopleth], transform=ORIG_PROJ)
    return im


def _coordinate_transform(x, y):
    return TARGET_PROJ.transform_points(ORIG_PROJ, x, y)


def calc_area(grid_lon, grid_lat, density, isopleth, area_threshold, **kwargs):
    contours_combined = get_contour(grid_lon, grid_lat, density, isopleth, **kwargs)
    contours_separate = separate_paths(contours_combined)
    areas = []
    polygons = []
    for region in contours_separate:
        trans_coords = _coordinate_transform(
            x=region.vertices[:, 0], y=region.vertices[:, 1]
        )
        x = trans_coords[:, 0]
        y = trans_coords[:, 1]
        a = 0.5 * np.sum(
            (y[:-1] * np.diff(x)) - (x[:-1] * np.diff(y))
        )  # calculate area in m^2
        a = np.abs(a) / (1000.0**2)  # convert to km^2
        if a >= area_threshold:
            poly = Polygon(
                [(i, j) for i, j in zip(region.vertices[:, 0], region.vertices[:, 1])]
            )
            areas.append(a)
            polygons.append(poly)
    return areas, polygons
    
def separate_paths(contour, **kwargs):	
	paths_by_layer = []
	for i, joined_paths_in_layer in enumerate(contour.get_paths()):
		separated_paths_in_layer = []
		path_vertices = []
		path_codes = []
		for verts, code in joined_paths_in_layer.iter_segments():
			if code == Path.MOVETO:
				if path_vertices:
					separated_paths_in_layer.append(Path(np.array(path_vertices), np.array(path_codes)))
				path_vertices = [verts]
				path_codes = [code]
			elif code == Path.LINETO:
					path_vertices.append(verts)
					path_codes.append(code)
			elif code == Path.CLOSEPOLY:
					path_vertices.append(verts)
					path_codes.append(code)
		if path_vertices:
			separated_paths_in_layer.append(Path(np.array(path_vertices), np.array(path_codes)))
		paths_by_layer.append(separated_paths_in_layer)
	return separated_paths_in_layer

def _mask_outside_region(lon, lat, polygon):
    return shapely.vectorized.contains(geometry=polygon, x=lon, y=lat)

def calc_polygon_statistics(lons, lats, spi_drought, spi_pluvial, polygon): #whiplash_points - not sure I need
	"""
    Function to calculate the following statistics for a given event polygon:
        1. Drought SPI (area_averaged)
        2. Pluvial SPI (area_averaged)
        3. SPI Change (area_averaged)
        4. Drought SPI (max - magnitude only)
        5. Pluvial SPI (max - magnitude only)
	6. SPI Change (max_change)

    Parameters
    ----------
    lon : numpy.ndarray, type float
        2-D array of longitudes
    lat : numpy.ndarray, type float
        2-D array of latitudes
    spi_drought : numpy.ndarray, type float
        2-D array of SPI during drought month. Should have dimensionality (lat, lon).
    spi_pluvial : numpy.ndarray, type float
        2-D array of SPI during pluvial month. Should have dimensionality (lat, lon).qa	
    whiplash_points : numpy.ndarray, type bool
        2-D boolean array specifying which grid points were labelled as having a whiplash occur
    polygon : shapely.geometry.Polygon
        Shapely polygon for the event.

    Returns
    -------
    Dictionary containing the 6 statistics given above; keys are drought_spi_area_avg, pluvial_spi_area_avg, spi_change_area_avg,
    drought_spi_max, pluvial_spi_max, and spi_change_max.
	"""
    # mask to only event region
	mask = _mask_outside_region(lons, lats, polygon)
	spi_drought_new = np.ma.masked_array(spi_drought, ~mask)
	spi_pluvial_new = np.ma.masked_array(spi_pluvial, ~mask)
   
    # mask any nans that slipped through
	spi_drought_new = np.ma.masked_invalid(spi_drought_new)  
	spi_pluvial_new = np.ma.masked_invalid(spi_pluvial_new) 
    
	# area-avg spi
	weights = np.cos(np.radians(lats[:, 0]))
	drought_tmp = np.ma.mean(spi_drought_new, axis=-1)  # avg across lons first
	pluvial_tmp = np.ma.mean(spi_pluvial_new, axis=-1)  # avg across lons first
    
	drought_spi_area_avg = np.ma.average(drought_tmp, weights=weights)
	pluvial_spi_area_avg = np.ma.average(pluvial_tmp, weights=weights)
	spi_change_area_avg = pluvial_spi_area_avg - drought_spi_area_avg
  
	drought_spi_max = np.ma.min(spi_drought_new) # Drought SPI (max - magnitude only) 
	pluvial_spi_max = np.ma.max(spi_pluvial_new) # Pluvial SPI (max - magnitude only) 
	
	spi_change_max = np.ma.max(spi_pluvial_new - spi_drought_new)
    
	return dict(
		drought_spi_area_avg=drought_spi_area_avg,
        pluvial_spi_area_avg=pluvial_spi_area_avg,
        spi_change_area_avg=spi_change_area_avg,
        drought_spi_max=drought_spi_max,
        pluvial_spi_max=pluvial_spi_max,
        spi_change_max=spi_change_max,
        	)

def linreg(x, y, min_lag, max_lag):
	# Initialize matrices
	slopes = []
	yintercepts = []
	rvalues = []
	pvalues = []
	stderrors = []
	# How to disypher what lag relationships mean:
		# Negative lag implies that x at the moment correlates with your y in the future
		# Positive lag implies that x at the moment correlates with you y in the past
	for lag in tqdm(range(min_lag,max_lag+1)):	# itterate from min_lag to max_lag (+1 because of range function)
		if lag == 0:	
			slope, yintercept, rvalue, pvalue, stderror = scs.linregress(x, y)
		elif lag < 0:
			slope, yintercept, rvalue, pvalue, stderror = scs.linregress(x[:lag], y[-lag:])
		elif lag > 0:
			slope, yintercept, rvalue, pvalue, stderror = scs.linregress(x[lag:], y[:-lag])
		
		# Append the values!
		slopes.append(slope)
		yintercepts.append(yintercept)
		rvalues.append(rvalue)
		pvalues.append(pvalue)
		stderrors.append(stderror)

	# Compile data into a dataarray for easy management
	print('Starting Compilation')
	da_reg = xr.DataArray(
		data=np.arange(min_lag,max_lag+1),
		dims='lag',
		coords=dict(
			slope = ('lag', slopes),
			yintercept = ('lag', yintercepts),
			rvalue = ('lag', rvalues),
			pvalue = ('lag', pvalues),
			stderror= ('lag',stderrors)
			),
		name='lin_reg'
		)

	return da_reg
	
def pearsons_corr(x, y, min_lag, max_lag, alternative_hypothesis):
	# Initialize matrices
	corr_coef = []
	pvalues = []
	# How to disypher what lag relationships mean:
		# Negative lag implies that x at the moment correlates with your y in the future
		# Positive lag implies that x at the moment correlates with you y in the past
	for lag in tqdm(range(min_lag,max_lag+1)):	# itterate from min_lag to max_lag (+1 because of range function)
		if lag == 0:	
			stat, pvalue = scs.pearsonr(x, y, alternative=str(alternative_hypothesis))
		elif lag < 0:
			stat, pvalue = scs.pearsonr(x[:lag], y[-lag:], alternative=str(alternative_hypothesis))
		elif lag > 0:
			stat, pvalue = scs.pearsonr(x[lag:], y[:-lag], alternative=str(alternative_hypothesis))
		
		# Append the values!
		corr_coef.append(stat)
		pvalues.append(pvalue)

	# Compile data into a dataarray for easy management
	print('Starting Compilation')
	da_reg = xr.DataArray(
		data=np.arange(min_lag,max_lag+1),
		dims='lag',
		coords=dict(
			corr_coef = ('lag', corr_coef),
			pvalue = ('lag', pvalues),
			),
		name='pearsons_corr_coeff'
		)

	return da_reg
	
def subset_events(df, overlap_days):
	subset_index = []
	subset_index.append(0)
	first_column_name = df.columns[0]
	first_column = df.iloc[:, 0]
	for i in range(0, len(first_column)-1):
		target_row = df.iloc[i]
		overlap_date = (pd.to_datetime(target_row[first_column_name]).date() + timedelta(days=overlap_days)).strftime('%Y-%m-%d')
		if first_column[i+1] <= overlap_date:
			subset_index.append(i)
		else:
			break
	
	subset_index = list(set(subset_index))
	subset = df.iloc[subset_index]
	polys = [shapely.wkt.loads(i) for i in subset.geometry]
	
	num_regions = len(subset)

	lat = np.arange(25.15625, 50.03125, 0.0625)
	lon = np.arange(235.40625, 293.03125, 0.0625) #Livneh Grid
	grid_lons, grid_lats = np.meshgrid(lon,lat)
	
	regions_masked = np.zeros((num_regions, lat.size, lon.size))
	
	for i,p in enumerate(polys):
		regions_masked[i,:,:] = _mask_outside_region(lon=grid_lons, lat=grid_lats, polygon=p)
		
	n,y,x = regions_masked.shape
	regions_masked = regions_masked.reshape(n,y*x)
	return regions_masked.astype(int)

def group_events(arr, thresh):
    coefs = np.zeros((arr.shape[0]))
    for i in range(coefs.shape[0]):
    	coefs[i], _ = scs.pearsonr(x=arr[0,:], y=arr[i,:])
    iloc = np.where(coefs >= thresh)[0]
    return coefs, iloc	
    
def start_dates(df):
	subset_begin_ind = np.where((df.Day_No == 0))[0]
	dates = df[df.columns[4]].iloc[subset_begin_ind].reset_index(drop=True)
	
	return dates
	
def end_dates_and_polys(df):
	end_dates = []
	polygons = []
	for i in tqdm(range(0,np.nanmax(df.Event_No))):
		subset = df.iloc[np.where((df.Event_No == i+1))[0]]
		end_dates.append(df[df.columns[4]].iloc[subset.Day_No.idxmax()])
		polygons.append(shapely.wkt.loads(df.geometry.iloc[subset.Area.idxmax()]))
		
	return end_dates, polygons 

def QuantileRegression(quantiles, df, data):
	X = sm.add_constant(df['time']) # Add constant for intercept in regression
	models = {q: sm.QuantReg(data, X).fit(q=q) for q in quantiles} # Fit Quantile Regression models for different quantiles
	
	# Predict values for visualization
	x_range = np.linspace(df['time'].min(), df['time'].max(), 100)
	X_pred = sm.add_constant(x_range)

	predictions = {q: models[q].predict(X_pred) for q in quantiles}

	#Find the slope of the lines
	slopes = {p: round(scs.linregress(x_range, predictions[p]).slope, 3) for p in predictions}
	
	return x_range, predictions, slopes

