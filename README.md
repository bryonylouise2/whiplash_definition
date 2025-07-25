Precipitation whiplash events have been increasingly studied in the scientific literature over the recent decade. However, no studies have considered the spatial coherence of grid points in their examination of whiplash events. Therefore, we developed an algorithm that defines spatially coherent precipitation whiplash events on the subseasonal-to-seasonal timescale between 1915 and 2020. We define precipitation whiplash events to occur when the 30-day Standardized Precipitation Index (SPI) moves from at or above +1 to at or below -1, or vice versa. Once grid points were identified as having undergone a whiplash occurrence, Kernel Density Estimation was used to create event polygons, taking into consideration the continuity of grid points. Events were then clustered into geographical regions using k-means clustering, allowing for the examination of their climatology and characteristics across the continental United States (CONUS). The developed definition is designed in such a way that it can be adapted, and the databases generated can be used in future studies to greatly enhance the understanding of precipitation whiplash events across the CONUS.

For more information and details regarding the event identification, please see the journal article:  Puxley, B. L. and Martin, E. R. (2025) ‘A Continental United States Climatology of Precipitation Whiplash Using a New Event-Based Definition’ - In Review.

For access to the complete databases for drought-to-pluvial, pluvial-to-drought, drought, and pluvial events, please see the .csv files: Puxley, B. L. and Martin, E. R. (2025) ‘A Continental United States Climatology of Precipitation Whiplash Using a New Event-Based Definition’. Zenodo. doi: 10.5281/zenodo.16414184.

Any use of the programs in this GitHub repository should cite the journal article Puxley, B. L. and Martin, E. R. (2025). Any use of the databases in the Zenodo repository should cite the journal article and the Zenodo repository.

# whiplash_definition

Order of Files
1. Augment_PRISM_Livneh.py
		This file combines the Livneh (1915-2012) and PRISM (2012-2020) datasets by bilinearly interpolating the PRISM precipitation dataset from its native 4 km grid onto Livneh's grid of about 6 km using the Python library xESMF (Zhuang et al. 2020).
		Type: Function
		Input: Precipitation data
		Output: netCDF file of combined Livneh and PRISM precipitation at ~ 6 km grid resolution from 1915 to 2020

2. CalculatingSPI.py
 		This file calculates the Standardized Precipitation Index (SPI) across the CONUS from 1915 to 2020. SPI can either be computed for CONUS completely or split into 12 different regions to save on memory. This file will also output a plot of the SPI, allowing for visualization of the data. Choose from 30-, 60-, 90-, or 180- days.
   		Input: Precipitation Data
		Output: netCDF file of rolling SPI values (choose from 30-, 60-, 90-, or 180- days) at a ~ 6 km grid resolution from 1915 to 2020; and a PNG file of the average SPI across the chosen region for visualization.

3. Convert_regional_spi_files_to_decadal_files.py
   		This file combines the regional SPI.nc files (if each region were run separately) into a CONUS-wide dataset, which is then split into decadal periods for later analysis. Can be edited to convert a CONUS-wide file into decadal files.
   		Input: regional SPI Data
		Output: multiple netCDF files of rolling SPI values at a ~ 6 km grid resolution from 1915 to 2020, split into 10-year decade periods. If files were regional, they have been combined into CONUS-wide.
  
5. Whiplash_Indentification.py
		This file identifies whiplash occurrences for all grid points across the CONUS between 1915-2020. This file will also output a plot of the whiplash count at each individual grid point across the region.
		Input: NetCDF file of SPI data.
    Output: 

6. Spatial_Consistency.py
7. Density_Calculation.py
8. Area_Calculation.py
9. Make_Area_Plot.py
10. Database_Creation.py
11. Independence_Algorithm.py
12. Clustering_Events.py
     
