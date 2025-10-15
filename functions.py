#########################################################################################
## File of functions needed and used for database creation
## Bryony Louise Puxley
## Last Edited: Friday, August 8th, 2025
#########################################################################################
#Import Required Modules
#########################################################################################
import gzip
import math
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from shapely.geometry import Polygon
import shapely.vectorized
from tqdm import tqdm
import scipy.stats as scs
import statsmodels.api as sm
from sklearn.utils import resample
from sklearn.linear_model import QuantileRegressor
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
#Kernel Density Estimation - used in 6.Spatial_Consistency.py to define large-scale
# events considering spatial consistency
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

#Calculate area of polygons - used in 8.Area_Calculation.py
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

# A function to separate multiple polygons that are classified as one polygon into their separate polygons.
# Not used in code - another method was found.
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

# A function to mask the region outside of a polygon
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

def calc_polygon_statistics_drought_pluvial_only(lons, lats, spi, polygon, max_process): #whiplash_points - not sure I need
	"""
    Function to calculate the following statistics for a given event polygon:
        1. SPI (area_averaged)
        2. SPI (max - magnitude only)

    Parameters
    ----------
    lon : numpy.ndarray, type float
        2-D array of longitudes
    lat : numpy.ndarray, type float
        2-D array of latitudes
    spi : numpy.ndarray, type float
        2-D array of SPI during specified month. Should have dimensionality (lat, lon).
    polygon : shapely.geometry.Polygon
        Shapely polygon for the event.

    Returns
    -------
    Dictionary containing the 6 statistics given above; keys are drought_spi_area_avg, pluvial_spi_area_avg, spi_change_area_avg,
    drought_spi_max, pluvial_spi_max, and spi_change_max.
	"""
    # mask to only event region
	mask = _mask_outside_region(lons, lats, polygon)
	spi_new = np.ma.masked_array(spi, ~mask)
   
    # mask any nans that slipped through
	spi_new = np.ma.masked_invalid(spi_new)  

	# area-avg spi
	weights = np.cos(np.radians(lats[:, 0]))
	spi_tmp = np.ma.mean(spi_new, axis=-1)  # avg across lons first
    
	spi_area_avg = np.ma.average(spi_tmp[0], weights=weights)

	if max_process == 'drought':
		spi_max = np.ma.min(spi_new) # Drought SPI (max - magnitude only) 
	elif max_process == 'pluvial':
		spi_max = np.ma.max(spi_new) # Pluvial SPI (max - magnitude only) 
	
    
	return dict(
		spi_area_avg=spi_area_avg,
        spi_max=spi_max,
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
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    subset_index = [0]  # always include the first row
    last_included_date = pd.to_datetime(df.iloc[0, 0]).date()

    for i in range(1, len(df)):
        current_date = pd.to_datetime(df.iloc[i, 0]).date()
        if (current_date - last_included_date).days <= overlap_days:
            subset_index.append(i)
        else:
            break  # stop as soon as the gap exceeds overlap_days

    subset = df.iloc[subset_index].copy()
    polys = [shapely.wkt.loads(i) for i in subset.geometry]

    num_regions = len(subset)
    lat = np.arange(25.15625, 50.03125, 0.0625)
    lon = np.arange(235.40625, 293.03125, 0.0625)  # Livneh Grid
    grid_lons, grid_lats = np.meshgrid(lon, lat)

    regions_masked = np.zeros((num_regions, lat.size, lon.size))

    for i, p in enumerate(polys):
        regions_masked[i, :, :] = _mask_outside_region(lon=grid_lons, lat=grid_lats, polygon=p)

    n, y, x = regions_masked.shape
    regions_masked = regions_masked.reshape(n, y * x)
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
	
def start_dates_droughts_pluvials(df):
	subset_begin_ind = np.where((df.Day_No == 0))[0]
	dates = df[df.columns[2]].iloc[subset_begin_ind].reset_index(drop=True)
	
	return dates
	
def end_dates_and_polys(df):
	end_dates = []
	polygons = []
	for i in tqdm(range(0,np.nanmax(df.Event_No))):
		subset = df.iloc[np.where((df.Event_No == i+1))[0]]
		end_dates.append(df[df.columns[4]].iloc[subset.Day_No.idxmax()])
		polygons.append(shapely.wkt.loads(df.geometry.iloc[subset.Area.idxmax()]))
		
	return end_dates, polygons 
	
def end_dates_and_polys_droughts_pluvials(df):
	end_dates = []
	polygons = []
	for i in tqdm(range(0,np.nanmax(df.Event_No))):
		subset = df.iloc[np.where((df.Event_No == i+1))[0]]
		end_dates.append(df[df.columns[2]].iloc[subset.Day_No.idxmax()])
		polygons.append(shapely.wkt.loads(df.geometry.iloc[subset.Area.idxmax()]))
		
	return end_dates, polygons 

#Weak (0.5-0.9); Moderate (1.0-1.4), Strong (1.5-1.9), Very Strong (>=2.0)
def categorize_enso_strength(value):
	if 1.0 > value >= 0.5:
		return 1
	if 1.5 > value >= 1.0:
		return 2
	if 2.0 > value >= 1.5:
		return 3
	if value >=2.0:
		return 4
	if -0.5 >= value > -1.0:
		return -1
	if -1.0 >= value > -1.5:
		return -2
	if -1.5 >= value > -2.0:
		return -3
	if value <=-2.0:
		return -4
	else:
		return 0

def categorize_enso_events(df, indices, value, min_consecutive, threshold, col):
    """Assigns a value if at least min_consecutive values are in the range."""
    start = None
    for i in range(len(indices)):
        if start is None:
            start = indices[i]
        if i == len(indices) - 1 or indices[i+1] != indices[i] + 1:
            if indices[i] - start + 1 >= min_consecutive:
            	df.loc[start:indices[i], col] = value
            start = None

#sklearn - must faster than statsmodel
def QuantileRegression(x, y, quantiles):
	#sklearn
	x = np.array(x)
	y = np.array(y)
    
	models = {q: QuantileRegressor(quantile=q, alpha=0) for q in quantiles}
	{q: models[q].fit(x.reshape(-1, 1), y) for q in quantiles}
	
	predictions = {q:  models[q].predict(x.reshape(-1, 1))for q in quantiles}
	slopes = {q: round(models[q].coef_[0],3) for q in quantiles}
	
	return predictions, slopes

def spans_zero(value1, value2):
    return value1 * value2 > 0

#sklearn - must faster than statsmodel
def bca_bootstrap(x, y, quantiles, niter, alpha):
    """
    Perform BCa bootstrap for quantile regression slope significance testing.
    
    Parameters:
    x (array-like): Independent variable.
    y (array-like): Dependent variable.
    quantile (float): Quantile level (e.g., 0.1, 0.5, 0.9).
    niter (int): Number of bootstrap resamples.
    alpha (float): Significance level
    
    Returns:
    tuple: (original slope, lower CI, upper CI, p-value)
    """
    x = np.array(x)
    y = np.array(y)
    models = {q: QuantileRegressor(quantile=q, alpha=0) for q in quantiles}
    {q: models[q].fit(x.reshape(-1, 1), y) for q in quantiles}
    
    orig_slopes = {q: round(models[q].coef_[0],3) for q in quantiles}
    
    lower_bounds = {}
    upper_bounds = {}
    pvalues = {}
    sig = {}
    linestyle = {}
    
    for q in quantiles:
    	boot_slopes = []
    	for _ in tqdm(range(niter)):
    		idx = resample(range(len(y)), replace=True)
    		x_boot = x[idx]
    		y_boot = y[idx]
    		boot_model = QuantileRegressor(quantile=q, alpha=0)
    		boot_model.fit(x_boot.reshape(-1, 1), y_boot)
    		
    		boot_slopes.append(boot_model.coef_[0])
    
    	boot_slopes = np.array(boot_slopes)
    
    	# BCa confidence intervals
    	z0 = scs.norm.ppf((boot_slopes < orig_slopes[q]).mean())
    	
    	if not np.isfinite(z0):
    		print(f"Warning: z0 is infinite for quantile {q}, skipping BCa and p-value.")
    		lower_bounds[q] = np.nan
    		upper_bounds[q] = np.nan
    		pvalues[q] = np.nan
    		sig[q] = np.nan
    		continue
    	
    	else:
    		a = (np.sum((np.mean(boot_slopes) - boot_slopes)**3) /
    				(6 * (np.sum((np.mean(boot_slopes) - boot_slopes)**2)**(3/2))))
    				
    		alpha_1 = scs.norm.cdf(z0 + (z0 + scs.norm.ppf(alpha / 2)) / (1 - a * (z0 + scs.norm.ppf(alpha / 2))))
    		alpha_2 = scs.norm.cdf(z0 + (z0 + scs.norm.ppf(1 - alpha / 2)) / (1 - a * (z0 + scs.norm.ppf(1 - alpha / 2))))
    		
    		lower_bounds[q] = round(np.percentile(boot_slopes, 100 * alpha_1),3)
    		upper_bounds[q] = round(np.percentile(boot_slopes, 100 * alpha_2),3)
    		
    		# P-value calculation (two-tailed test against null hypothesis of no trend)
    		pvalues[q] = round(2 * min((boot_slopes > 0).mean(), (boot_slopes < 0).mean()),3)
    		
    		sig[q] = spans_zero(lower_bounds[q], upper_bounds[q])
    		
    		if sig[q] == True:
    			linestyle[q] = '-'
    		else:
    			linestyle[q] = '--'
    	
    return orig_slopes, lower_bounds, upper_bounds, pvalues, sig, linestyle

def permutation_test(data1, data2, num_permutations=10000):
    observed_diff = round(np.mean(data2) - np.mean(data1),3)
       
    combined_data = np.concatenate([data1, data2])  # Merge both groups
    
    count = 0
    for _ in range(num_permutations):
    	np.random.shuffle(combined_data)  # Shuffle the data
    		
    	# Split the shuffled data into two new random groups
    	perm_group1 = np.array(combined_data[:len(data1)])
    	perm_group2 = np.array(combined_data[len(data1):])
    		
    	perm_diff = np.mean(perm_group2) - np.mean(perm_group1) # Compute new difference
    		
    	if abs(perm_diff) >= abs(observed_diff):  # Check if permuted diff is at least as extreme
    		count += 1
	
    p_value = round(count / num_permutations,3)  # Compute p-value
    	
    return observed_diff, p_value
    
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    r = 6371  # Radius of earth in kilometers. Use 3959 for miles
    return c * r

def plot_largest_polygon(ax, polygons, colors, **kwargs):
    # Flatten all geometries into a list of Polygons
    all_polys = []
    for polygon in polygons:
        if polygon.geom_type == 'MultiPolygon':
            all_polys.extend(polygon.geoms)
        else:
            all_polys.append(polygon)
    
    # Find the largest polygon by area
    largest_poly = max(all_polys, key=lambda p: p.area)
    
    # Plot the largest polygon
    x, y = largest_poly.exterior.xy
    ax.plot(x, y, transform=ccrs.PlateCarree(), color=color, **kwargs)
    
    # Get centroid and add label
    centroid = largest_poly.centroid
    cx, cy = centroid.x, centroid.y
    
    ax.text(cx, cy, str(i+1), transform=ccrs.PlateCarree(),
            ha='center', va='center', fontsize=10,
            fontweight='bold', color=color,
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1', alpha=1))		

def polygon_area_km2(polygon, from_crs="EPSG:4326", to_crs="EPSG:6933"):
    """
    Compute area of a Shapely polygon in km^2 using projection + shoelace formula.
    polygon: shapely.geometry.Polygon
    from_crs: CRS of polygon coords (default EPSG:4326 = lon/lat)
    to_crs: equal-area CRS for projection (default EPSG:6933 = global equal area)
    """
    # Transformer
    transformer = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)
    
    # Get exterior coords
    coords = np.array(polygon.exterior.coords)  # shape (N, 2)
    lon, lat = coords[:, 0], coords[:, 1]
    
    # Project to meters
    x, y = transformer.transform(lon, lat)
    
    # Shoelace formula
    a = 0.5 * np.sum((y[:-1] * np.diff(x)) - (x[:-1] * np.diff(y)))
    
    return abs(a) / 1e6  # km²
