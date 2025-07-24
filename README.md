Precipitation whiplash events have been increasingly studied in the scientific literature over the recent decade. However, no studies have considered the spatial coherence of grid points in their examination of whiplash events. Therefore, we developed an algorithm that defines spatially coherent precipitation whiplash events on the subseasonal-to-seasonal timescale between 1915 and 2020. We define precipitation whiplash events to occur when the 30-day Standardized Precipitation Index (SPI) moves from at or above +1 to at or below -1, or vice versa. Once grid points were identified as having undergone a whiplash occurrence, Kernel Density Estimation was used to create event polygons, taking into
consideration the continuity of grid points. Events were then clustered into geographical regions using k-means clustering, allowing for the examination of their climatology and characteristics across the continental United States (CONUS).

Journal Article:  Puxley, B. L. and Martin, E. R. (2025) ‘A Continental United States Climatology of Precipitation Whiplash Using a New Event-Based Definition’ - In Review

Complete Databases: Puxley, B. L. and Martin, E. R. (2025) ‘A Continental United States Climatology of Precipitation Whiplash Using a New Event-Based Definition’. Zenodo. doi: 10.5281/zenodo.16414184.

# whiplash_definition

Order of Files
1. Augment_PRISM_Livneh.py
		This file combines the Livneh (1915-2012) and PRISM (2012-2020) datasets by bilineraly interpolating the PRISM precipitation dataset from its native 4 km grid onto Livneh's grid of about 6 km using python library xESMF (Zhuang et al. 2020).
		Type: Function
		Input: Precipitation data

2. CalculatingSPI.py
 		This file calculates the Standardized Precipitation Index (SPI) across the CONUS between 1915-2020. SPI can either be computed for CONUS completely or split into 12 different regions to save on memory. This file will also output a plot of the SPI to allow for a visualization of the data.
   	Output: netCDF file of SPI data and a .png plot of the SPI data.

3. Whiplash_Indentification.py
		This file identifies whiplash occurrences for all grid points across the CONUS between 1915-2020. This file will also output a plot of the whiplash count at each individual grid point across the region.
		Input: NetCDF file of SPI data.
    Output: 
