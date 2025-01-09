#########################################################################################
## File of functions needed and used for database creation
## Bryony Louise
## Last Edited: Wednesday January 8th 2025
#########################################################################################
#Import Required Modules
#########################################################################################
import gzip
import numpy as np
from sklearn.neighbors import KernelDensity
from shapely.geometry import Polygon
import shapely.vectorized

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
    
	drought_avg = np.ma.average(drought_tmp, weights=weights)
	pluvial_avg = np.ma.average(pluvial_tmp, weights=weights)
    
	#drought_spi_area_avg = 
	#pluvial_spi_area_avg =  
	#spi_change_area_avg = pluvial_spi_area_avg - drought_spi_area_avg
  
	drought_spi_max = np.ma.min(spi_drought_new) # Drought SPI (max - magnitude only) 
	pluvial_spi_max = np.ma.max(spi_pluvial_new) # Pluvial SPI (max - magnitude only) 
    
	return dict(
		drought_spi_area_avg,
        pluvial_spi_area_avg,
        spi_change_area_avg,
        drought_spi_max,
        pluvial_spi_max,
        spi_change_max
        	)
