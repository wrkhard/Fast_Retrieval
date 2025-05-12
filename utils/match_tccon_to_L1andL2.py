import numpy as np
import h5py as h5
import pickle
import glob


def match_L1_on_soundingID(tccon_dict, l1bsc_dict, bands_dict):

    # calculate wls
    o2_coefs = l1bsc_dict['InstrumentHeader']['dispersion_coef_samp'][0,:,:]
    wco2_coefs = l1bsc_dict['InstrumentHeader']['dispersion_coef_samp'][1,:]
    sco2_coefs = l1bsc_dict['InstrumentHeader']['dispersion_coef_samp'][2,:]


    # o2a_fp0_coefs = l1['InstrumentHeader']['dispersion_coef_samp'][0,0,:]

    # term_1 = (o2a_fp0_coefs[0] ** 1) * (i_n ** 0)
    # term_2 = (o2a_fp0_coefs[1] ** 2) * (i_n ** 1)
    # term_3 = (o2a_fp0_coefs[2] ** 3) * (i_n ** 2)
    # term_4 = (o2a_fp0_coefs[3] ** 4) * (i_n ** 3)
    # term_5 = (o2a_fp0_coefs[4] ** 5) * (i_n ** 4)   
    # term_6 = (o2a_fp0_coefs[5] ** 6) * (i_n ** 5)

    # wls = term_1 + term_2 + term_3 + term_4 + term_5 + term_6

    indexes = np.arange(1, 1017, 1)


    o2_wls = np.zeros((8, 1016))
    wco2_wls = np.zeros((8, 1016))
    sco2_wls = np.zeros((8, 1016))

    for fp in range(8):
        print(fp)
        o2_coefs_fp = o2_coefs[fp,:]
        wco2_coefs_fp = wco2_coefs[fp,:]
        sco2_coefs_fp = sco2_coefs[fp,:]
        o2_wls[fp, :] = 0
        wco2_wls[fp, :] = 0
        sco2_wls[fp, :] = 0
        for k in range(1,7):
            o2_wls[fp, :] =  o2_wls[fp, :] + (o2_coefs_fp[k-1] * (indexes ** (k-1)))
            wco2_wls[fp, :] = wco2_wls[fp, :] + (wco2_coefs_fp[k-1] * (indexes ** (k-1)))
            sco2_wls[fp, :] = sco2_wls[fp, :] + (sco2_coefs_fp[k-1] * (indexes ** (k-1)))


    # print(f"o2_wls shape: {o2_wls.shape}")
    # print(f"wco2_wls shape: {wco2_wls.shape}")
    # print(f"sco2_wls shape: {sco2_wls.shape}")
    # print(f"o2_wls: {o2_wls}")
    # print(f"wco2_wls: {wco2_wls}")
    # print(f"sco2_wls: {sco2_wls}")

    matched_sid_coords = np.where(np.isin(l1bsc_dict['SoundingGeometry']['sounding_id'][:,:], tccon_dict['sounding_id_collocated']))
    if matched_sid_coords[0].size == 0:
        pass;
    # change from tuple to a list
    for i, j in zip(matched_sid_coords[0], matched_sid_coords[1]):
        row, footprint = int(i), int(j)
        # change the list to a numpy array
        matched_sid_coords = np.array([row, footprint])
        matched_sid = l1bsc_dict['SoundingGeometry']['sounding_id'][matched_sid_coords[0], matched_sid_coords[1],]

        # get the radiances for the matched index for each band
        o2 = l1bsc_dict['SoundingMeasurements']['radiance_o2'][matched_sid_coords[0], matched_sid_coords[1],]
        wco2 = l1bsc_dict['SoundingMeasurements']['radiance_weak_co2'][matched_sid_coords[0], matched_sid_coords[1],]
        sco2 = l1bsc_dict['SoundingMeasurements']['radiance_strong_co2'][matched_sid_coords[0], matched_sid_coords[1],]

        bands_dict['o2_band_obs'].append(o2)
        bands_dict['wco2_band_obs'].append(wco2)
        bands_dict['sco2_band_obs'].append(sco2)

        # append the meta data
        bands_dict['sounding_id'].append(matched_sid)

        # get bad_pixel mask and coefs for calculating the wls
        bands_dict['bad_pixels_o2'].append(l1bsc_dict['InstrumentHeader']['bad_sample_list'][0,footprint,:])
        bands_dict['bad_pixels_wco2'].append(l1bsc_dict['InstrumentHeader']['bad_sample_list'][1,footprint,:])
        bands_dict['bad_pixels_sco2'].append(l1bsc_dict['InstrumentHeader']['bad_sample_list'][2,footprint,:])

        bands_dict['o2_wls'].append(o2_wls[footprint,:])
        bands_dict['wco2_wls'].append(wco2_wls[footprint,:])
        bands_dict['sco2_wls'].append(sco2_wls[footprint,:])       

        


    return bands_dict

def match_L2_on_soundingID(colloc_dict, oco_dict, combined_dict):
    matches_in_colloc = np.in1d(colloc_dict['sounding_id_collocated'], oco_dict['RetrievalHeader/sounding_id'])
    matches_in_oco2 = np.in1d(oco_dict['RetrievalHeader/sounding_id'], colloc_dict['sounding_id_collocated'])


    # add the matched tccoon collocation data to the combined_dict first
    for key in colloc_dict:
        matched_data = colloc_dict[key][matches_in_colloc]
        # append each value into the combined_dict
        for item in matched_data:
            combined_dict[key].append(item)

    # add the matched oco data to the combined_dict
    for key in oco_dict:
        matched_data = oco_dict[key][matches_in_oco2]
        # append each value into the combined_dict
        for item in matched_data:
            combined_dict[key].append(item)

    # print the number of matches we have for the file
    if np.sum(matches_in_colloc) > 0:
        print(f"Number of matches: {np.sum(matches_in_colloc)}")




    return combined_dict

    



def main(years=['2015','2016','2017','2018','2019','2020','2021']):

    STATE_VARS = [
                'RetrievalResults/surface_pressure_apriori_fph',
                'PreprocessingResults/surface_pressure_apriori_abp', 
                'RetrievalResults/outcome_flag',
                'RetrievalResults/xco2', 
                'RetrievalResults/xco2_uncert',
                ]    
    GEOM_VARS = [
                'RetrievalGeometry/retrieval_solar_zenith',
                'RetrievalGeometry/retrieval_solar_distance',
                'RetrievalGeometry/retrieval_solar_zenith', 
                'RetrievalGeometry/retrieval_zenith',
                'RetrievalGeometry/retrieval_solar_azimuth', 
                'RetrievalGeometry/retrieval_azimuth', 
                ]
    META_VARS = [
                'RetrievalHeader/sounding_id', 
                'RetrievalGeometry/retrieval_longitude', 
                'RetrievalGeometry/retrieval_latitude',
                ]
    BAND_VARS = [
                'SoundingGeometry/sounding_id',
                'SoundingMeasurements/radiance_o2',
                'SoundingMeasurements/radiance_weak_co2',
                'SoundingMeasurements/radiance_strong_co2',
                 ]

    TCCON_VARS = ['sounding_id_collocated', 'xco2tccon', 'xco2bc', 'xco2_uncertainty_collocated', 'xco2_quality_flag']
    
    L2STD_VARS = STATE_VARS + GEOM_VARS + META_VARS


    # open the tccon npz for the year
    for year in years:
        print(f"Processing year {year}")
        year_data = {}
        bands_data = {}
        for key in TCCON_VARS:
            year_data[key] = []
        for key in L2STD_VARS:
            year_data[key] = []

        bands_data = {'o2_band_obs' : [], 'wco2_band_obs' : [], 'sco2_band_obs' : [],  'sounding_id' : [], 'bad_pixels_o2' : [], 'bad_pixels_wco2' : [], 'bad_pixels_sco2' : [], 'o2_wls' : [], 'wco2_wls' : [], 'sco2_wls' : []}

        tccon_data = np.load(f'/scratch-science/algorithm/wkeely/Data/TCCON_GGG20_ACOS_v11_year{year}.npz')
        tccon_dict = {}
        for key in tccon_data.files:
            tccon_dict[key] = tccon_data[key]
            if key == 'sounding_id_collocated':
                bands_data[key] = tccon_data[key] # add the "collocated" sounding_id to the bands_data dictionary to check that we have matched the correct sounding_id

        # TODO Glob the L2Std files /oco2/product/Obs_B110008_r01/year/month/day/L2Std/oco2_L2StdND_*.h5
        l2std_files = glob.glob(f'/oco2/product/Ops_B11*/{year}/*/*/L2Std/oco2_L2StdND_*.h5')
        print(f"Found {len(l2std_files)} L2Std files for {year}")
        for l2std_file in l2std_files:
            # open the L2Std file
            with h5.File(l2std_file, 'r') as f:
                l2std_dict = {}
                for key in L2STD_VARS:
                    groups = key.split('/')
                    l2std_dict[key] = f[groups[0]][groups[1]]
                # compare the tccon and L2Std files
                year_data = match_L2_on_soundingID(tccon_dict, l2std_dict, year_data)

        print(f"Number of matches for the year: {len(year_data['RetrievalHeader/sounding_id'])}")
        # convert to np arrays
        for key in year_data:
            year_data[key] = np.array(year_data[key])
            year_data[key] = np.expand_dims(year_data[key], axis=1)
        
        # save the year_data to a pickle file
        with open(f'/scratch-science/algorithm/wkeely/Data/L2Std_TCCON_Lnd_matched_{year}.pkl', 'wb') as f:
            pickle.dump(year_data, f)

        del year_data # clean up

        l1bsc_files = glob.glob(f'/oco2/product/Ops_B11*/{year}/*/*/L1bSc/oco2_L1bScND_*.h5')
        # sort the file
        l1bsc_files.sort()
        print(f"Files found: {l1bsc_files}")

        for l1bsc_file in l1bsc_files:
            # open the L1BSc file
            with h5.File(l1bsc_file, 'r') as f:
                print(f"Currently looking in file: {l1bsc_file} ...")
                l1bsc_dict = {}
                # for key in BAND_VARS:
                    # groups = key.split('/')
                    # print(f"groups: {groups}")
                    # l1bsc_dict[key] = f[groups[0]][groups[1]]
                    # l1bsc_dict[key] = f[key]
                # compare the tccon and L2Std files
                bands_data = match_L1_on_soundingID(tccon_dict, f, bands_data)
        # stack band data
        for key in bands_data:
            bands_data[key] = np.vstack(bands_data[key])
        
        print(f"Number of matches for the year: {len(bands_data['sounding_id'])}")

        # save the bands_data to a pickle file
        with open(f'/scratch-science/algorithm/wkeely/Data/L1bSc_TCCON_Lnd_matched_{year}.pkl', 'wb') as f:
            pickle.dump(bands_data, f)
        
        del bands_data # clean up

    



if __name__ == "__main__":
    main()
