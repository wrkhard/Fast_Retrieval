import pandas as pd
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import h5py as h5
import glob
import tqdm
import multiprocessing as mp

from scipy.interpolate import interp1d


def update_num_colors(num_colors, sample_idxs):
    for p in range(len(sample_idxs)):
        diff = np.diff(sample_idxs[p])
        # get the indicies from sample_idxs where the difference is less than zero
        new_colors = np.where(diff < 0)
        # convert idx to list
        new_colors = list(new_colors[0])
        # add one to each idx
        new_colors = [i+1 for i in new_colors]
        new_colors[2] = new_colors[2] - new_colors[1]
        new_colors[1] = new_colors[1] - new_colors[0]

        num_colors[p] = new_colors
    
    return num_colors

def remove_eofs(year = 2015, pickle_path = '/Users/williamkeely/Desktop/Fast_Retrieval/Data/', save_path = '/Users/williamkeely/Desktop/Fast_Retrieval/Data/'):
    pickle_path = pickle_path + 'L2DiaND_XCO2_' + str(year) + '_downsampled.p'
    year_pickle = pickle.load(open(pickle_path, "rb"))

    EOF_DIR = '/Users/williamkeely/Desktop/Fast_Retrieval/eofs/'
    EOF_FILE = 'OCO2_qtsb11_eofs_00000-40000_oceanG_alt2_falt1_L2.h5'
    eof_file = h5.File(EOF_DIR + EOF_FILE, 'r')

    ref_obs = year_pickle['ref_obs']
    ref_sims = year_pickle['ref_sim']
    uncerts = year_pickle['uncerts']
    sample_idxs = year_pickle['sample_idxs']
    sample_idxs = sample_idxs.astype(int)
    num_colors = year_pickle['num_color']
    wls = year_pickle['wl']
    states = year_pickle['state']
    state_names = year_pickle['state_var']
    num_colors = update_num_colors(num_colors, sample_idxs)

    eof_1_o2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Land/EOF_1_waveform_1'][:]
    eof_2_o2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Land/EOF_2_waveform_1'][:]
    eof_3_o2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Land/EOF_3_waveform_1'][:]

    # EOFs for WCO2 band for each footprint
    eof_1_wco2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Land/EOF_1_waveform_2'][:]
    eof_2_wco2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Land/EOF_2_waveform_2'][:]
    eof_3_wco2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Land/EOF_3_waveform_2'][:]

    # EOFs for SCO2 band for each footprint
    eof_1_sco2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Land/EOF_1_waveform_3'][:]
    eof_2_sco2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Land/EOF_2_waveform_3'][:]
    eof_3_sco2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Land/EOF_3_waveform_3'][:]

    # get the footprint which is the last digit of the sounding id

    sid = states[:, state_names.index('RetrievalHeader/sounding_id')]
    footprints = sid % 10
    footprints = footprints.astype(int)
    footprints = footprints - 1

    # get the scaling factors for each eof
    eof_scale_1_1 = states[:, state_names.index('RetrievalResults/eof_1_scale_o2')]
    eof_scale_1_2 = states[:, state_names.index('RetrievalResults/eof_2_scale_o2')]
    eof_scale_1_3 = states[:, state_names.index('RetrievalResults/eof_3_scale_o2')]

    eof_scale_2_1 = states[:, state_names.index('RetrievalResults/eof_1_scale_weak_co2')]
    eof_scale_2_2 = states[:, state_names.index('RetrievalResults/eof_2_scale_weak_co2')]
    eof_scale_2_3 = states[:, state_names.index('RetrievalResults/eof_3_scale_weak_co2')]

    eof_scale_3_1 = states[:, state_names.index('RetrievalResults/eof_1_scale_strong_co2')]
    eof_scale_3_2 = states[:, state_names.index('RetrievalResults/eof_2_scale_strong_co2')]
    eof_scale_3_3 = states[:, state_names.index('RetrievalResults/eof_3_scale_strong_co2')]

    # ref_sims_no_eof = []
    o2_band = []
    o2_band_OLD = []
    sco2_band = []
    wco2_band = []

    o2_band_no_eof = [] 
    sco2_band_no_eof = []
    wco2_band_no_eof = []

    # obs 
    o2_band_obs = []
    sco2_band_obs = []
    wco2_band_obs = []

    # sample indicies
    o2_band_idxs = []
    o2_band_idxs_OLD = []
    sco2_band_idxs = []
    wco2_band_idxs = []

    # wl
    o2_wl = []
    o2_wl_OLD = []
    sco2_wl = []
    wco2_wl = []

    for idx, fp in enumerate(footprints):
        padding = np.zeros(np.sum(ref_sims[idx]==-9.9999900e+05))
        padding.fill(-9.9999900e+10) # padding will return to e+05 after subtracting off eof

        # O2A band
        x = eof_1_o2[fp,sample_idxs[idx,:num_colors[idx,0]]] * uncerts[idx,:num_colors[idx,0]]
        s11 = 1e+19 / np.sqrt(sum((x - np.mean(x))**2)/(1016-1))
        t11 = (x * s11)
        term_o2_1 = t11  * eof_scale_1_1[idx]

        x = eof_2_o2[fp,sample_idxs[idx,:num_colors[idx,0]]] * uncerts[idx,:num_colors[idx,0]]
        s12 = 1e+19 / np.sqrt(sum((x - np.mean(x))**2)/(1016-1))
        t12 = (x * s12)
        term_o2_2 = t12 * eof_scale_1_2[idx]

        x = eof_3_o2[fp,sample_idxs[idx,:num_colors[idx,0]]] * uncerts[idx,:num_colors[idx,0]]
        s13 = 1e+19 / np.sqrt(sum((x - np.mean(x))**2)/(1016-1))
        t13 = (x * s13)
        term_o2_3 = t13 * eof_scale_1_3[idx]

        # terms to concatenate for o2a band.
        term1 = (term_o2_1+term_o2_2+term_o2_3)

        #WCO2 band
        x = eof_1_wco2[fp,sample_idxs[idx,num_colors[idx,0]:num_colors[idx,0]+num_colors[idx,1]]] * uncerts[idx,num_colors[idx,0]:num_colors[idx,0]+num_colors[idx,1]]
        s21 = 1e+19 / np.sqrt(sum((x - np.mean(x))**2)/(1016-1))
        t21 = (x * s21)
        term_wco2_1 = t21 * eof_scale_2_1[idx]

        x = eof_2_wco2[fp,sample_idxs[idx,num_colors[idx,0]:num_colors[idx,0]+num_colors[idx,1]]] * uncerts[idx,num_colors[idx,0]:num_colors[idx,0]+num_colors[idx,1]]
        s22 = 1e+19 / np.sqrt(sum((x - np.mean(x))**2)/(1016-1))
        t22 = (x * s22)
        term_wco2_2 = t22 * eof_scale_2_2[idx]

        x = eof_3_wco2[fp,sample_idxs[idx,num_colors[idx,0]:num_colors[idx,0]+num_colors[idx,1]]] * uncerts[idx,num_colors[idx,0]:num_colors[idx,0]+num_colors[idx,1]]
        s23 = 1e+19 / np.sqrt(sum((x - np.mean(x))**2)/(1016-1))
        t23 = (x * s23)
        term_wco2_3 = t23 * eof_scale_2_3[idx]

        # terms to concatenate for wco2 band.
        term2 = (term_wco2_1+term_wco2_2+term_wco2_3)
        
        #SCO2 band
        x = eof_1_sco2[fp,sample_idxs[idx,num_colors[idx,0]+num_colors[idx,1]:num_colors[idx,0]+num_colors[idx,1]+num_colors[idx,2]]] * uncerts[idx,num_colors[idx,0]+num_colors[idx,1]:num_colors[idx,0]+num_colors[idx,1]+num_colors[idx,2]]
        s31 = 1e+19 / np.sqrt(sum((x - np.mean(x))**2)/(1016-1))
        t31 = (x * s31)
        term_sco2_1 = t31 * eof_scale_3_1[idx]

        x = eof_2_sco2[fp,sample_idxs[idx,num_colors[idx,0]+num_colors[idx,1]:num_colors[idx,0]+num_colors[idx,1]+num_colors[idx,2]]] * uncerts[idx,num_colors[idx,0]+num_colors[idx,1]:num_colors[idx,0]+num_colors[idx,1]+num_colors[idx,2]]
        s32 = 1e+19 / np.sqrt(sum((x - np.mean(x))**2)/(1016-1))
        t32 = (x * s32)
        term_sco2_2 = t32 * eof_scale_3_2[idx]

        x = eof_3_sco2[fp,sample_idxs[idx,num_colors[idx,0]+num_colors[idx,1]:num_colors[idx,0]+num_colors[idx,1]+num_colors[idx,2]]] * uncerts[idx,num_colors[idx,0]+num_colors[idx,1]:num_colors[idx,0]+num_colors[idx,1]+num_colors[idx,2]]
        s33 = 1e+19 / np.sqrt(sum((x - np.mean(x))**2)/(1016-1))
        t33 = (x * s33)
        term_sco2_3 = t33 * eof_scale_3_3[idx]

        # terms to concatenate for sco2 band.
        term3 = (term_sco2_1+term_sco2_2+term_sco2_3)

        
        # concatenate the terms
        # scaled_eof = np.concatenate((term1,term2,term3,padding),axis=0)
        o2_band.append(ref_sims[idx,:num_colors[idx,0]])
        wco2_band.append(ref_sims[idx,num_colors[idx,0]:num_colors[idx,0]+num_colors[idx,1]])
        sco2_band.append(ref_sims[idx,num_colors[idx,0]+num_colors[idx,1]:num_colors[idx,0]+num_colors[idx,1]+num_colors[idx,2]])

        o2_band_no_eof.append(ref_sims[idx,:num_colors[idx,0]] - term1)
        wco2_band_no_eof.append(ref_sims[idx,num_colors[idx,0]:num_colors[idx,0]+num_colors[idx,1]] - term2)
        sco2_band_no_eof.append(ref_sims[idx,num_colors[idx,0]+num_colors[idx,1]:num_colors[idx,0]+num_colors[idx,1]+num_colors[idx,2]] - term3)

        o2_band_obs.append(ref_obs[idx,:num_colors[idx,0]])
        wco2_band_obs.append(ref_obs[idx,num_colors[idx,0]:num_colors[idx,0]+num_colors[idx,1]])
        sco2_band_obs.append(ref_obs[idx,num_colors[idx,0]+num_colors[idx,1]:num_colors[idx,0]+num_colors[idx,1]+num_colors[idx,2]])

        o2_band_idxs.append(sample_idxs[idx,:num_colors[idx,0]])
        wco2_band_idxs.append(sample_idxs[idx,num_colors[idx,0]:num_colors[idx,0]+num_colors[idx,1]])
        sco2_band_idxs.append(sample_idxs[idx,num_colors[idx,0]+num_colors[idx,1]:num_colors[idx,0]+num_colors[idx,1]+num_colors[idx,2]])

        o2_wl.append(wls[idx,:num_colors[idx,0]])
        wco2_wl.append(wls[idx,num_colors[idx,0]:num_colors[idx,0]+num_colors[idx,1]])
        sco2_wl.append(wls[idx,num_colors[idx,0]+num_colors[idx,1]:num_colors[idx,0]+num_colors[idx,1]+num_colors[idx,2]])

    # save the pickle file with the band_wl, band_idxs, band_no_eof, band_obs, and band
    save_path = save_path + 'L2DiaND_XCO2_' + str(year) + '_downsampled_eof_removed.p'
    save_pickle = {'o2_band': o2_band, 
                   'wco2_band': wco2_band, 
                   'sco2_band': sco2_band, 
                   'o2_band_no_eof': o2_band_no_eof, 
                   'wco2_band_no_eof': wco2_band_no_eof, 
                   'sco2_band_no_eof': sco2_band_no_eof, 
                   'o2_band_obs': o2_band_obs, 
                   'wco2_band_obs': wco2_band_obs, 
                   'sco2_band_obs': sco2_band_obs, 
                   'o2_band_idxs': o2_band_idxs, 
                   'wco2_band_idxs': wco2_band_idxs, 
                   'sco2_band_idxs': sco2_band_idxs, 
                   'o2_wl': o2_wl, 
                   'wco2_wl': wco2_wl, 
                   'sco2_wl': sco2_wl,
                   'num_color': num_colors,
                   'states': states,
                   'state_var': state_names
                   }
    pickle.dump(save_pickle, open(save_path, "wb"))
    print('done')


def change_radiance_gridding(band = "o2", year = 2015, pickle_path = '/Users/williamkeely/Desktop/Fast_Retrieval/Data/'):
    pickle_path = pickle_path + 'L2DiaND_XCO2_' + str(year) + '_downsampled_eof_removed.p'
    year_pickle = pickle.load(open(pickle_path, "rb"))

    if band == "o2":
        band = year_pickle['o2_band']
        band_no_eof = year_pickle['o2_band_no_eof']
        band_obs = year_pickle['o2_band_obs']
        band_idxs = year_pickle['o2_band_idxs']
        band_wl = year_pickle['o2_wl']
        pixel_lut = np.arange(91, 920, 1)
        grid_start_stop = (0.7592, 0.7715)
        cut_from_front_back = (10,-20)
        pixel_lut = pixel_lut.astype(int)
    elif band == "wco2":
        band = year_pickle['wco2_band']
        band_no_eof = year_pickle['wco2_band_no_eof']
        band_obs = year_pickle['wco2_band_obs']
        band_idxs = year_pickle['wco2_band_idxs']
        band_wl = year_pickle['wco2_wl']
        pixel_lut = np.arange(211, 863, 1)
        grid_start_stop = (1.598, 1.6177)
        cut_from_front_back = (20, -20)
        pixel_lut = pixel_lut.astype(int)
    elif band == "sco2":
        band = year_pickle['sco2_band']
        band_no_eof = year_pickle['sco2_band_no_eof']
        band_obs = year_pickle['sco2_band_obs']
        band_idxs = year_pickle['sco2_band_idxs']
        band_wl = year_pickle['sco2_wl']
        pixel_lut = np.arange(100, 911, 1)
        grid_start_stop = (2.047, 2.0796)
        cut_from_front_back = (20, -20)
        pixel_lut = pixel_lut.astype(int)
    else:
        print('band string not recognized ... should be "o2", "wco2", or "sco2"')
        return

    # find and fill missing sample indicies
    for p in range(len(band_idxs)):
        # find values missing from band_idxs
        missing = np.setdiff1d(pixel_lut,band_idxs[p], )
        # print('missing = ', missing)
        for m in missing:  
            # get the index of the missing values
            idx = np.where(np.in1d(pixel_lut, m))
            # insert missing value into the o2_band_idxs
            band_idxs[p] = np.insert(band_idxs[p], idx[0], m)
            # insert np.nan into o2_band_no_eof and o2_band_obs at the same index
            band_no_eof[p] = np.insert(band_no_eof[p], idx[0], np.nan)
            band_obs[p] = np.insert(band_obs[p], idx[0], np.nan)
            band_wl[p] = np.insert(band_wl[p], idx[0], np.nan)
            band[p] = np.insert(band[p], idx[0], np.nan)


    # interpolate and align to common gridding
    x= np.linspace(grid_start_stop[0],grid_start_stop[1], band_no_eof[0].shape[0])
    x = x.astype(float)

    band_no_eof_interp = []
    for p in range(len(band_no_eof)):
        f = interp1d(band_wl[p], band_no_eof[p], kind='linear', fill_value='extrapolate')
        band_no_eof_interp.append(f(x))
        f = interp1d(band_wl[p], band_obs[p], kind='linear', fill_value='extrapolate')
        band_obs[p] = f(x)
        f = interp1d(band_wl[p], band[p], kind='linear', fill_value='extrapolate')
        band[p] = f(x)

    # chop ends off
    for p in range(len(band_no_eof_interp)):
        band_no_eof_interp[p] = band_no_eof_interp[p][cut_from_front_back[0]:cut_from_front_back[1]] # 10:-20 for o2a band, 20:-20 for wco2 band, 20:-20 for sco2 band
        band_obs[p] = band_obs[p][cut_from_front_back[0]:cut_from_front_back[1]]
        band[p] = band[p][cut_from_front_back[0]:cut_from_front_back[1]]
        band_idxs[p] = band_idxs[p][cut_from_front_back[0]:cut_from_front_back[1]]
        band_wl[p] = x[cut_from_front_back[0]:cut_from_front_back[1]]
        
    
    

    return band, band_no_eof, band_obs, band_idxs, band_wl





    
    


    






    
