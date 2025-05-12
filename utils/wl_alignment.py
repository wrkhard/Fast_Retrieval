import numpy as np
import pickle
import glob


# William Keely william.r.keely [at] jpl.nasa.gov; Steffen Mauceri steffen.mauceri [at] jpl.nasa.gov
# Script to align the wavelengths of the reflectance simulations and reflectance observations

# TODO : add auto encoder test pickle output

import numpy as np
from scipy.interpolate import interp1d

def align_to_grid(data, band_key, wls_key, idxs_key, pixel_lut_range, cut_from_front_back):
    # Get the band, wavelength, and index data
    band = data[band_key]
    band_wl = data[wls_key]
    band_idxs = data[idxs_key]
    
    # Define pixel lookup table and grid start/stop
    pixel_lut = np.arange(*pixel_lut_range).astype(int)
    grid_start_stop = (band_wl[0][0], band_wl[0][-1])

    # Align data to the common grid by inserting missing indices
    for p in range(len(band)):
        missing = np.setdiff1d(pixel_lut, band_idxs[p])
        for m in missing:
            idx = int(np.where(pixel_lut == m)[0][0])
            band_idxs[p] = np.insert(band_idxs[p], idx, m)
            band[p] = np.insert(band[p], idx, np.nan)
            band_wl[p] = np.insert(band_wl[p], idx, np.nan)

    # Interpolate the bands
    x = np.linspace(grid_start_stop[0], grid_start_stop[1], band[0].shape[0]).astype(float)
    band_interp = []
    for p in range(len(band)):
        f = interp1d(band_wl[p], band[p], kind="linear", fill_value="extrapolate")
        band_interp.append(f(x))

    # Cut off the ends as per the provided indices
    for p in range(len(band_interp)):
        band_interp[p] = band_interp[p][cut_from_front_back[0]:cut_from_front_back[1]]
        band_wl[p] = x[cut_from_front_back[0]:cut_from_front_back[1]]
        band_idxs[p] = band_idxs[p][cut_from_front_back[0]:cut_from_front_back[1]]

        # Update the data dictionary
        data[band_key][p] = band_interp[p]
        data[wls_key][p] = band_wl[p]
        data[idxs_key][p] = band_idxs[p]


""" Example!!! """
# Now you can call this function for each band with the corresponding parameters
# align_to_grid(data, "o2_band", "o2_wls", "o2_idxs", (91, 950), (10, -129))
# align_to_grid(data, "wco2_band", "wco2_wls", "wco2_idxs", (211, 900), (20, -157))
# align_to_grid(data, "sco2_band", "sco2_wls", "sco2_idxs", (100, 950), (70, -148))



    
