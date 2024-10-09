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
