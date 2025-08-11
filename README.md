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
   		Input: regional SPI data
		Output: multiple netCDF files of rolling SPI values at a ~ 6 km grid resolution from 1915 to 2020, split into 10-year decade periods. If files were regional, they have been combined into a CONUS-wide file.
  
4. Whiplash_Indentification.py
		This file identifies whiplash occurrences for all grid points across the CONUS from 1915 to 2020. This file will also output a plot of the whiplash count at each grid point across the region.
		Input: Regional SPI files or CONUS-wide SPI file - need all times at each grid point.
		Output: netCDF file of identified whiplash occurrences at a ~ 6 km grid resolution from 1915 to 2020, and a PNG file of the average SPI across the chosen region for visualization.

5. Convert_regional_whiplash_files_to_decadal_files.py
   		This file combines the regional whiplash.nc files (if each region were run separately) into a CONUS-wide dataset, which is then split into decadal periods for later analysis. Can be edited to convert a CONUS-wide file into decadal files.
   		Input: regional whiplash data
		Output: multiple netCDF files of identified whiplash occurrences at a ~ 6 km grid resolution from 1915 to 2020, split into 10-year decade periods. If files were regional, they have been combined into a CONUS-wide file.

6. Spatial_Consistency.py
   		This file utilizes Kernel Density Estimation (KDE) to assess the spatial continuity of grid points experiencing a whiplash event and to determine the corresponding events.
		Input: Decadal whiplash occurrence files.
		Output: netCDF files of the density of whiplash grid points at each day throughout the period. Higher values represent a greater density of whiplash occurrences.

7. Density_Calculation.py
   		This file combines the multiple density.nc files into one file and calculates the 99th percentile, which is used to draw the polygons outlining the area defined as having a whiplash event.
		Input: decadal density files.
		Output: The 99th percentile of densities for drought-to-pluvial and pluvial-to-drought events to be used to draw the single contour for the polygon outlining the area defined as having a whiplash event.
 
8. Area_Calculation.py
		This file calculates the area of all polygons generated by the KDE process and outputs CSV files of potential events.
		Input: Decadal density files.
		Output: Decadal CSV files (1 for drought-to-pluvial, 1 for pluvial-to-drought) that have a list of all potential events throughout the time frame, including: Drought Date, Pluvial Date, Whiplash Date, Area (km2), and polygon geometry. 

9. Database_Creation.py
		This script creates the databases of all of the precipitation whiplash events, subsetting by areal threshold (can be set to zero to keep all events), and also calculates relevant statistics such as: the area-averaged SPI during the 30-day drought period, the area-averaged SPI during the 30-day pluvial period, the area-averaged SPI change, the grid point magnitude maximum SPI during the 30-day drought and pluvial periods, respectively, and the grid point maximum SPI change.
		Input: Decadal SPI files, decadal whiplash files, decadal density/KDE files, and decadal potential events CSV files.
		Output: Two CSV files of drought-to-pluvial and pluvial-to-drought events.

10. Independence_Algorithm.py
		This script contains an objective post-processing algorithm to group "repeat" events,
		Input: Subsetted by area event file from 9. Database_Creation.py
		Output: A CSV file of either independent drought-to-pluvial or independent pluvial-to-drought events.

11. Determine_Clusters.py
    		This script determines the appropriate number of clusters to group events into regions across the CONUS (k-means clustering). This script uses the Elbow Method (Thorndike, 1953) and the Silhouette Coefficient (Rousseeuw, 1987).
		Input: Independent event files of Drought-to-Pluvial and Pluvial-to-Drought events.
		Output: PNG files. 1) The results of the Elbow Method, Silhouette Coefficient, Events retained, and Hybrid Index to help determine the appropriate number of clusters. 2) The average polygons of all the events in each cluster for the entire range of k-values to view for spatial coherence, and 3) Supplementary Figure 4 from the journal article.

12. Event_Cluster_Assignment.py
    		This script clusters events into regions across the CONUS (k-means clustering) based on the appropriate number of clusters determined in 11.Determine_Clusters.py. The script uses the largest areal overlap (intersection area) to assign events to clusters.

# Supplementary/Extra Files
functions.py
	 		This file contains most of the functions required for database creation.

8.5. Make_Area_Plot.py
			This file plots the distribution of all the areas calculated in 8.Area_Calculation for examination and shows where the chosen area threshold sits.   
     		Input: The CSV files of all potential events created in 8.Area_Calculation.py.
			Output: A PNG file of a 4-panel plot that includes a histogram of the frequency of all areas, and a line plot of the percentiles relating to each area for both types of precipitation whiplash events.

10.5. SPI_autocorrelation.py
			This script determines the temporal lag autocorrelation of SPI across the CONUS
			Input: Either decadal SPI files or a time-only-averaged (averaged across lat and lon) SPI file.
   			Output: A PNG file of the temporal lag autocorrelation of SPI (CONUS averaged) 

Ex.CaseStudyAnalysis_DP.py
			This script examines and creates GIFs of specific drought-to-pluvial events
			Input: Decadal SPI, whiplash occurrences, normalized density, and independent drought-to-pluvial events CSV. Choose event number or date.
			Output: GIFs of SPI during the drought period, SPI during the pluvial period, SPI change, and whiplash occurrences.

Ex.CaseStudyAnalysis_PD.py 
				This script examines and creates GIFs of specific pluvial-to-drought events
				Input: Decadal SPI, whiplash occurrences, normalized density, and independent pluvial-to-drought events CSV. Choose event number or date.
				Output: GIFs of SPI during the drought period, SPI during the pluvial period, SPI change, and whiplash occurrences.


Ex.Number_of_Days_Between_NonIndependent_Events.py
			This script calculates and plots the days between non-independent dates within the database.
			Input: Subsetted by area event file from 9. Database_Creation.py
   			Output: A histogram of a) the number of days between drought-to-pluvial and b) pluvial-to-drought events.

Ex.Event_Characteristics.py
			This script calculates and plots on what "day" the largest area, the largest area-averaged, and the largest max SPI change occur. 
			Input: Independent event files of Drought-to-Pluvial and Pluvial-to-Drought events.
			Output: A PNG file of on what "day" does the largest area, the largest area-averaged, and the max SPI change.  

 
# Analysis Files
A.Event_Climatology.py
			This script creates a map of the event climatology across the CONUS, i.e., the total number of times and the most common month that a grid point is within an extreme-event polygon over the period 1915-2020
			Input: Independent event files of Drought-to-Pluvial, Pluvial-to-Drought, Drought, and Pluvial events.
			Output: Two PNG files. Figure 5 from the journal article: grid point frequency for the period 1915-2020 for a) drought-to-pluvial, b) pluvial-to-drought, c) drought, and d) pluvial events. Figure 6 from the journal article: The most common season that a grid point experiences a) drought-to-pluvial, b) pluvial-to-drought, c) drought, and d) pluvial events.




