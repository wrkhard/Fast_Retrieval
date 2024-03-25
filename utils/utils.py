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
    band_rows = []
    for p in range(len(sample_idxs)):
        try:
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
        except:
            print('error with num_colors at index ', p)
            band_rows.append(p)
            continue

    
    return num_colors, band_rows

def remove_eofs(year = 2015, pickle_path = '/scratch-science/algorithm/wkeely/', save_path = '/scratch-science/algorithm/wkeely/'):
    pickle_path = pickle_path + 'L2DiaND_XCO2_' + str(year) + '_downsampled.p'
    year_pickle = pickle.load(open(pickle_path, "rb"))

    # EOF_DIR = '/Users/williamkeely/Desktop/Fast_Retrieval/eofs/'
    # EOF_FILE = 'OCO2_qtsb11_eofs_00000-40000_oceanG_alt2_falt1_L2.h5'
    # eof_file = h5.File(EOF_DIR + EOF_FILE, 'r')
    EOF_DIR = '/home/wkeely/OCO_L2DIA/EOF/'
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
    num_colors, bad_rows = update_num_colors(num_colors, sample_idxs) # use correct sample indicies to remove eofs

    # remove any bad rows
    ref_obs = np.delete(ref_obs, bad_rows, axis=0)
    ref_sims = np.delete(ref_sims, bad_rows, axis=0)
    uncerts = np.delete(uncerts, bad_rows, axis=0)
    wls = np.delete(wls, bad_rows, axis=0)
    states = np.delete(states, bad_rows, axis=0,)
    sample_idxs = np.delete(sample_idxs, bad_rows, axis=0)
    num_colors = np.delete(num_colors, bad_rows, axis=0)


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
    sco2_band_idxs = []
    wco2_band_idxs = []

    # wl
    o2_wl = []
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


def common_radiance_gridding(band = "o2", year = 2015, pickle_path = '/scratch-science/algorithm/wkeely/'):
    pickle_path = pickle_path + 'L2DiaND_XCO2_' + str(year) + '_downsampled_eof_removed.p'
    year_pickle = pickle.load(open(pickle_path, "rb"))

    if band == "o2":
        band = year_pickle['o2_band']
        band_no_eof = year_pickle['o2_band_no_eof']
        band_obs = year_pickle['o2_band_obs']
        band_idxs = year_pickle['o2_band_idxs']
        band_wl = year_pickle['o2_wl']
        pixel_lut = np.arange(91, 950, 1)
        grid_start_stop = (0.7592, 0.7715)
        cut_from_front_back = (10,-129)
        pixel_lut = pixel_lut.astype(int)
    elif band == "wco2":
        band = year_pickle['wco2_band']
        band_no_eof = year_pickle['wco2_band_no_eof']
        band_obs = year_pickle['wco2_band_obs']
        band_idxs = year_pickle['wco2_band_idxs']
        band_wl = year_pickle['wco2_wl']
        pixel_lut = np.arange(211, 900, 1)
        grid_start_stop = (1.598, 1.6177)
        cut_from_front_back = (20, -157)
        pixel_lut = pixel_lut.astype(int)
    elif band == "sco2":
        band = year_pickle['sco2_band']
        band_no_eof = year_pickle['sco2_band_no_eof']
        band_obs = year_pickle['sco2_band_obs']
        band_idxs = year_pickle['sco2_band_idxs']
        band_wl = year_pickle['sco2_wl']
        pixel_lut = np.arange(100, 950, 1)
        grid_start_stop = (2.045, 2.0796)
        cut_from_front_back = (70, -148)
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

    # check for nan values and fill with interpolation
    
        
    
    

    return band, band_no_eof_interp, band_obs, band_idxs, band_wl



def denoising_data(band = "o2", dir_path = '/Users/williamkeely/Desktop/Fast_Retrieval/Data/', train_years = [2020], val_year = 2020, save_data = False):
    # load the data
    train_data = []
    val_data = []
    for year in train_years:
        pickle_path = dir_path + 'L2DiaND_XCO2_' + str(year) + '_downsampled_eof_removed_aligned.p'
        year_pickle = pickle.load(open(pickle_path, "rb"))
        if band == "o2":
            train_data.extend([(year_pickle['o2_band_no_eof'][i], year_pickle['o2_band_obs'][i]) for i in range(len(year_pickle['num_color']))])
        elif band == "wco2":
            train_data.extend([(year_pickle['wco2_band_no_eof'][i], year_pickle['wco2_band_obs'][i]) for i in range(len(year_pickle['num_color']))])
        elif band == "sco2":
            train_data.extend([(year_pickle['sco2_band_no_eof'][i], year_pickle['sco2_band_obs'][i]) for i in range(len(year_pickle['num_color']))])

    pickle_path = dir_path + 'L2DiaND_XCO2_' + str(val_year) + '_downsampled_eof_removed_aligned.p'
    year_pickle = pickle.load(open(pickle_path, "rb"))
    if band == "o2":
        val_data.extend([(year_pickle['o2_band_no_eof'][i], year_pickle['o2_band_obs'][i]) for i in range(len(year_pickle['num_color']))])
    elif band == "wco2":
        val_data.extend([(year_pickle['wco2_band_no_eof'][i], year_pickle['wco2_band_obs'][i]) for i in range(len(year_pickle['num_color']))])
    elif band == "sco2":
        val_data.extend([(year_pickle['sco2_band_no_eof'][i], year_pickle['sco2_band_obs'][i]) for i in range(len(year_pickle['num_color']))])

    if save_data:
        save_path = dir_path + 'denoising_data_' + band + '.p'
        save_pickle = {'train_data': train_data, 'val_data': val_data}
        pickle.dump(save_pickle, open(save_path, "wb"))

    return train_data, val_data

def normalize_data(train_data, val_data):
    # get the mean and std for the training data
    train_data = np.array(train_data)
    mean_no_eof = np.mean(train_data[:,0])
    std_no_eof = np.std(train_data[:,0])
    mean_obs = np.mean(train_data[:,1])
    std_obs = np.std(train_data[:,1])

    # normalize the training data
    train_data[:,0] = (train_data[:,0] - mean_no_eof) / std_no_eof
    train_data[:,1] = (train_data[:,1] - mean_obs) / std_obs

    # normalize the validation data
    val_data = np.array(val_data)
    val_data[:,0] = (val_data[:,0] - mean_no_eof) / std_no_eof
    val_data[:,1] = (val_data[:,1] - mean_obs) / std_obs

    #TODO -- save to the normalization parameters to a json file

    return train_data, val_data, mean_no_eof, std_no_eof, mean_obs, std_obs

# TODO -- Randomly scale and add EOFs to the training data. Function to be used by the retrieval data loader.

def randomly_scale_and_add_EOFs():
    pass

# TODO -- Normalize training and validation data. Function to be used by the retrieval data loader and denoising data loader.
# TODO -- Get mertrics function for the retrieval model. Function to be used by the retrieval model.
# TODO -- Get metrics function for the denoising model. Function to be used by the denoising model. Reconstruction error and PSNR.

# *********** SUPER UNDER CONSTRUCTION ***********
# TODO -- Generate retrieval training, and test datasets.
# TODO -- swap out no_eof for either outputs from UNETs or randomly scale EOFs into training years.
def _generate_retrieval_data(training=[2014,2015,2016,2017,2018,2019],test=2020):
    # load the training data
    load_path = '/scratch-science/algorithm/wkeely/'
    o2_band_no_eof = []
    wco2_band_no_eof = []
    sco2_band_no_eof = []
    num_colors_train = []
    train_states = []
    o2_band_obs_test = []
    wco2_band_obs_test = []
    sco2_band_obs_test = []
    
    num_colors_test = []
    test_states = []

    for year in training:
        file = pickle.load(open(load_path + 'L2DiaND_XCO2_' + str(year) + '_downsampled_eof_removed_aligned.p', "rb"))
        o2_band_no_eof.append(file['o2_band_no_eof'])
        wco2_band_no_eof.append(file['wco2_band_no_eof'])
        sco2_band_no_eof.append(file['sco2_band_no_eof'])
        train_states.append(file['states'])
        num_colors_train.append(file['num_color'])
        train_state_names = file['state_var']
    
    # load the test data
    file = pickle.load(open(load_path + 'L2DiaND_XCO2_' + str(test) + '_downsampled_eof_removed_aligned.p', "rb"))
    o2_band_obs_test.append(file['o2_band_obs'])
    wco2_band_obs_test.append(file['wco2_band_no_eof'])
    sco2_band_no_eof_test.append(file['sco2_band_no_eof'])
    test_states.append(file['states'])
    num_colors_test.append(file['num_color'])
    test_state_names = file['state_var']

    # vstack the data
    o2_band_no_eof = np.vstack(o2_band_no_eof)
    wco2_band_no_eof = np.vstack(wco2_band_no_eof)
    sco2_band_no_eof = np.vstack(sco2_band_no_eof)
    num_colors_train = np.vstack(num_colors_train)
    train_states = np.vstack(train_states)
    o2_band_no_eof_test = np.vstack(o2_band_no_eof_test)
    wco2_band_no_eof_test = np.vstack(wco2_band_no_eof_test)
    sco2_band_no_eof_test = np.vstack(sco2_band_no_eof_test)
    num_colors_test = np.vstack(num_colors_test)
    test_states = np.vstack(test_states)

    # remove nans from training and test data
    for band in [o2_band_no_eof,wco2_band_no_eof,sco2_band_no_eof]:
        nan_rows = np.argwhere(np.isnan(band))
        o2_band_no_eof = np.delete(o2_band_no_eof, nan_rows, axis=0)
        wco2_band_no_eof = np.delete(wco2_band_no_eof, nan_rows, axis=0)
        sco2_band_no_eof = np.delete(sco2_band_no_eof, nan_rows, axis=0)
        num_colors_train = np.delete(num_colors_train, nan_rows, axis=0)
        train_states = np.delete(train_states, nan_rows, axis=0)

    for band in [o2_band_no_eof_test,wco2_band_no_eof_test,sco2_band_no_eof_test]:
        nan_rows = np.argwhere(np.isnan(band))
        o2_band_no_eof_test = np.delete(o2_band_no_eof_test, nan_rows, axis=0)
        wco2_band_no_eof_test = np.delete(wco2_band_no_eof_test, nan_rows, axis=0)
        sco2_band_no_eof_test = np.delete(sco2_band_no_eof_test, nan_rows, axis=0)
        num_colors_test = np.delete(num_colors_test, nan_rows, axis=0)
        test_states = np.delete(test_states, nan_rows, axis=0)


    # scale by 1E-19
    o2_band_no_eof = o2_band_no_eof * 1e-19
    wco2_band_no_eof = wco2_band_no_eof * 1e-19
    sco2_band_no_eof = sco2_band_no_eof * 1e-19
    o2_band_no_eof_test = o2_band_no_eof_test * 1e-19
    wco2_band_no_eof_test = wco2_band_no_eof_test * 1e-19
    sco2_band_no_eof_test = sco2_band_no_eof_test * 1e-19

    _bands = ['o2', 'wco2', 'sco2']
    _my = []
    _sy = []
    _maxy = []
    # standardize the data
    for y in [o2_band_no_eof,wco2_band_no_eof,sco2_band_no_eof]:
        _my.append(np.mean(y, axis=0))
        _sy.append(np.std(y, axis=0))
        _maxy.append(np.max(y, axis=0))

    o2_band_no_eof = (o2_band_no_eof - _my[0]) / (_sy[0] * _maxy[0])
    wco2_band_no_eof = (wco2_band_no_eof - _my[1]) / (_sy[1] * _maxy[1])
    sco2_band_no_eof = (sco2_band_no_eof - _my[2]) / (_sy[2] * _maxy[2])

    o2_band_no_eof_test = (o2_band_no_eof_test - _my[0]) / (_sy[0] * _maxy[0])
    wco2_band_no_eof_test = (wco2_band_no_eof_test - _my[1]) / (_sy[1] * _maxy[1])
    sco2_band_no_eof_test = (sco2_band_no_eof_test - _my[2]) / (_sy[2] * _maxy[2])

    # save the standardization parameters
    save_path = '/scratch-science/algorithm/wkeely/'
    save_pickle = {'band':_bands, 'mean': _my, 'std': _sy, 'max': _maxy}
    pickle.dump(save_pickle, open(save_path + 'standardization_parameters_for_retrieval.p', "wb"))

    # select the state variables that will be used for as features and as labels
    features_names = ['RetrievalResults/surface_pressure_apriori_fph',
                      ]
    label_names = []
    meta_names = ['RetrievalHeader/sounding_id',
                  'RetrievalGeometry/retrieval_longitude',  
                  'RetrievalGeometry/retrieval_latitude']





    






    

    






    
