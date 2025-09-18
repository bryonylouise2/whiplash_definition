Precipitation whiplash events have been increasingly studied in the scientific literature over the recent decade. However, no studies have considered the spatial coherence of grid points in their examination of whiplash events. Therefore, we developed an algorithm that defines spatially coherent precipitation whiplash events on the subseasonal-to-seasonal timescale between 1915 and 2020. We define precipitation whiplash events to occur when the 30-day Standardized Precipitation Index (SPI) moves from at or above +1 to at or below -1, or vice versa. Once grid points were identified as having undergone a whiplash occurrence, Kernel Density Estimation was used to create event polygons, taking into consideration the continuity of grid points. Events were then clustered into geographical regions using k-means clustering, allowing for the examination of their climatology and characteristics across the continental United States (CONUS). The developed definition is designed to be adaptable, allowing the generated databases to be utilized in future studies and significantly enhancing the understanding of precipitation whiplash events across CONUS.

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

Ex.Compare_PRISM_&_Livneh.py
			This script to answer Reviewer Question #1: Can you evaluate whether there are notable differences in values between Livneh and PRISM using overlapping time periods? I'd like a bit more background to confirm the assumption that there is no influence on the  whiplash calculations between these datasets.
			Input:  Daily precipitation data from Livneh and PRISM from 1981 to 2020.
			Output: Two PNG files. 1) a timeseries of a) 30-day rolling precipitation totals in mm for PRISM (blue) and Livneh (red) between 1981 and 2010, and b) the difference (PRISM - Livneh) in 30-day rolling precipitation totals in mm. and 2) a spatial plot of the average annual precipitation total in inches (1981-2020) for a) PRISM and b) Livneh.

Ex.Event_Characteristics.py
			This script calculates and plots on what "day" the largest area, the largest area-averaged, and the largest max SPI change occur. 
			Input: Independent event files of Drought-to-Pluvial and Pluvial-to-Drought events.
			Output: A PNG file of on what "day" does the largest area, the largest area-averaged, and the max SPI change.  

Ex.Number_of_Days_Between_NonIndependent_Events.py
			This script calculates and plots the days between non-independent dates within the database.
			Input: Subsetted by area event file from 9. Database_Creation.py
   			Output: A histogram of a) the number of days between drought-to-pluvial and b) pluvial-to-drought events.

Ex.check_precip_data.py
			This script checks the precipitation data to make sure it makes sense.
			Input: Precipitation Data
			Output: A PNG of the annual average precipitation across the chosen region. 

Ex.check_spi_data.py
			This script checks the SPI Data to make sure it makes sense
			Input: regional SPI data
			Output: A PNG of the annual average SPI across the CONUS.

Ex.plot_whiplash_data.py
			This script plots the number of whiplash occurrences at each grid point throughout the timeframe
			Input: regional whiplash data
			Output: A PNG of the number of whiplash occurrences throughout the period across the CONUS.
 
# Analysis Files
A.Event_Climatology.py
			This script creates a map of the event climatology across the CONUS, i.e., the total number of times and the most common month that a grid point is within an extreme-event polygon over the period 1915-2020
			Input: Independent event files of Drought-to-Pluvial, Pluvial-to-Drought, Drought, and Pluvial events.
			Output: Two PNG files. Figure 5 from the journal article: grid point frequency for the period 1915-2020 for a) drought-to-pluvial, b) pluvial-to-drought, c) drought, and d) pluvial events. Figure 6 from the journal article: The most common season that a grid point experiences a) drought-to-pluvial, b) pluvial-to-drought, c) drought, and d) pluvial events.

A.Event_Examination.py
			This script examines and plots specific events.
			Input: Independent event files of Drought-to-Pluvial and Pluvial-to-Drought events. Previously made SPI, whiplash occurrence, and normalized density files, 
			Output: Two PNG files. Figure 1 from the journal article: 30-day SPI during (a) the drought period, and (b) the following pluvial period. (c) Points flagged as having a whiplash event as per our definition (see the text for details). (d) Full KDE normalized density field using the Epanechnikov kernel and 0.02 bandwidth. In red on all subplots, the event polygon is drawn using the 0.4878 contour. Figure S1 from the journal article: the same for pluvial-to-drought.

A.Event_Frequency.py
			This script calculates the frequency of events throughout the timeframe, as well as seasonally throughout the year.
			Input: Independent event files of Drought-to-Pluvial and Pluvial-to-Drought events. ENSO (Oceanic Niño Index-ONI) data. 
			Output: PNG files. 1) Plot of the yearly ENSO - ONI Index from 1915 to 2020. 2) The yearly frequency of the events; the average areal size; and the total area impacted from 1915 to 2020 (individually). 3) the monthly frequency of the events, and the monthly average areal size. 4) The trends on each half of the database. 5) The yearly frequency of the events; the average areal size; and the total area impacted from 1915 to 2020 (one plot).

A.Event_Intensity.py
			This script examines the intensity of whiplash events.
			Input: Independent event files of Drought-to-Pluvial and Pluvial-to-Drought events.
			Output: A scatter plot (PNG) of average SPI change vs maximum grid point SPI change.

A.Event_Length_Analysis.py
			This script examines and plots how long an event was in drought/pluvial conditions prior to and after the event
			Input: Independent event files of Drought-to-Pluvial, Pluvial-to-Drought, Drought only, and Pluvial only events.
			Output: A scatter plot (PNG) of (a,c) Time in drought (SPI below zero) prior to vs time in pluvial (SPI above zero) after our recorded drought-to-pluvial whiplash events, and (b,d) time in pluvial (SPI above zero) prior to vs time in drought (SPI below zero) after our recorded pluvial-to-drought whiplash events (Figure 11) and the expanded full version (Supplementary Figure 7). Additionally, the values for the average number of days in drought (SPI below zero) and pluvial (SPI above zero) conditions for each of our defined clusters, as shown in Table 2.

A.Event_Length_Calculation.py
			This script to calculate how long an event was in drought/pluvial conditions prior to and after the event
			Input: Independent event files of Drought-to-Pluvial, Pluvial-to-Drought, Drought only, and Pluvial only events. Decadal SPI fiLes. 
			Output: Updated independent event files of Drought-to-Pluvial, Pluvial-to-Drought, Drought only, and Pluvial only events with columns for Time in *CONDITION* Before, and Time in *CONDITION* After for respective events.

A.Event_Propagation.py
			This script calculates how far the center point of the polygon is moving during each event (Event Propagation) and how the distance between the center point of the polygon and the boundary is changing during each event (Event Growth)
			Input: Independent event files of Drought-to-Pluvial, Pluvial-to-Drought, Drought only, and Pluvial only events.
			Output: A PNG of a scatter plot of centroid propagation vs event growth (Figure 12)

A.Regional_Analysis_Seasonality.py
			This script calculates the seasonality of drought-to-pluvial and pluvial-to-drought events throughout the timeframe for individual clusters.
			Input: Independent event files of Drought-to-Pluvial and Pluvial-to-Drought events. El Nino Southern Oscillation Southern Oscillation Index (SOI). Daily Precipitation Data from 1915-2020.
			Output: Six PNG files. 1) A PNG of the yearly ENSO index from 1915 to 2020. 2) A PNG of the temporal trends of drought-to-pluvial and pluvial-to-drought precipitation whiplash events across CONUS (Figure 7 in the manuscript). 3) Two PNGs of the temporal trends in the  50th percentile (Figure 8) and 90th percentiles (Supplementary Figure 5) of drought-to-pluvial and pluvial-to-drought precipitation whiplash events for each of the 7 clusters. 4) A PNG of the seasonal cycle of the frequency of drought-to-pluvial and pluvial-to-drought precipitation whiplash events (Figure 9) and finally 5) A PNG of the seasonal cycle of average areal size of event polygons for drought-to-pluvial and pluvial-to-drought precipitation whiplash (Figure 10).

A.Regional_Analysis_Seasonality_Droughts_Pluvials_Only.py
			This script calculatea the seasonality of drought and pluvial only events throughout the timeframe for individual clusters
			Input: Independent event files of Drought only and Pluvial only events. 
			Output: PNG Files of 1) the temporal trends of drought and pluvial precipitation whiplash events across CONUS, 2) Two PNGs of the temporal trends in the  50th percentile (Supplementary Figure 6) and 90th percentiles of drought and pluvial only events for each of the 7 clusters, and 3) the seasonal cycle of the frequency of drought and pluvial only events.
