import numpy as np
import h5py as h5
import pickle
import glob
from multiprocessing import Pool

def make_pool_map(years,months,days):
    pool_map = []
    if days is None:
        for year in years:
            for month in months:
                pool_map.append([year,month])
        return pool_map
    else: 
        for year in years:
            for month in months:
                for day in days:
                    pool_map.append([year,month,day])
        return pool_map

def main(pool_map,nadir=True,downsample=True,debug=False,):

    year = pool_map[0]
    month = pool_map[1]
    day = pool_map[2]

    if nadir:
        save_name = '/scratch-science/algorithm/wkeely/Data/L2DiaND_XCO2_' + year + '_'+month+'_'+day+'.p'
        files = glob.glob('/data/oco2/ops/product/Ops_B11006_r02/' + year + '/'+ month+'/'+day+'/L2Dia/oco2_L2DiaND*.h5')
        region = 'Land'
    else:
        save_name = '/scratch-science/algorithm/wkeely/Data/L2DiaGL_XCO2_' + year + '_'+month+'_'+day+'.p'
        files = glob.glob('/data/oco2/ops/product/Ops_B11006_r02/' + year + '/'+ month+'/'+day+'/L2Dia/oco2_L2DiaGL*.h5')
        region = 'Ocean'

    if debug:
        files = files[:5]
        
    ref_sim = 'SpectralParameters/modeled_radiance'
    ref_ob = 'SpectralParameters/measured_radiance'
    ob_uncert = 'SpectralParameters/measured_radiance_uncert'
    sample_idx = 'SpectralParameters/sample_indexes'
    num_color = 'SpectralParameters/num_colors_per_band'
    # co2_profile_ap = 'RetrievalResults/co2_profile_apriori'
    # co2_profile = 'RetrievalResults/co2_profile'
    # press_weight = 'RetrievalResults/xco2_pressure_weighting_function'

    var_names = ['RetrievalResults/surface_pressure_apriori_fph',
                'RetrievalResults/wind_speed_apriori', 'RetrievalResults/eof_1_scale_weak_co2', 'RetrievalResults/eof_1_scale_strong_co2',
                    'RetrievalResults/eof_1_scale_o2', 'RetrievalResults/eof_2_scale_weak_co2', 'RetrievalResults/eof_2_scale_strong_co2', 
                    'RetrievalResults/eof_2_scale_o2', 'RetrievalResults/eof_3_scale_weak_co2', 'RetrievalResults/eof_3_scale_strong_co2',
                    'RetrievalResults/eof_3_scale_o2', 'RetrievalGeometry/retrieval_solar_zenith', 'RetrievalGeometry/retrieval_zenith',
                        'RetrievalGeometry/retrieval_solar_azimuth', 'RetrievalGeometry/retrieval_azimuth', 'PreprocessingResults/surface_pressure_apriori_abp', 
                        'PreprocessingResults/dispersion_multiplier_abp', 'RetrievalHeader/sounding_id',
                        'RetrievalGeometry/retrieval_longitude', 'RetrievalGeometry/retrieval_latitude',
                            'RetrievalResults/xco2', 'RetrievalGeometry/retrieval_solar_zenith',
                    'RetrievalGeometry/retrieval_land_fraction', 'RetrievalGeometry/retrieval_solar_distance',
                    'PreprocessingResults/cloud_flag_idp', 'RetrievalResults/outcome_flag']


    print('Reading State Vectors')
    # # get radiance
    ref_sims = []
    ref_obs = []
    uncerts = []
    sample_idxs = []
    num_colors = []
    wls = []
    state_names = []
    states = []
    f_vars = [] 

    # press_weights = []
    co2_profile_aps = []
    co2_profiles = []
      

    for file in files:
        print(file)
        # try to open file and get variables

        f = h5.File(file, 'r')
        # get radiance
        r_sim = f.get(ref_sim)[()]
        r_ob = f.get(ref_ob)[()]
        r_uncert = f.get(ob_uncert)[()]
        r_sample_idx = f.get(sample_idx)[()]
        r_color = f.get(num_color)[()]
        # r_press_weight = f.get(press_weight)[()]
        # r_co2_profile_ap = f.get(co2_profile_ap)[()]
        # r_co2_profile = f.get(co2_profile)[()]



        # if len(r_sim) < 10:
        #     print('skipping because len(r_sim) = ' + str(len(r_sim)))
        #     continue
        # if len(r_ob) < 10:
        #     print('skipping because len(r_ob) = ' + str(len(r_ob)))
        #     continue
        # if r_sim.shape[1] >= 2300:  # 68
        #     print('skipping because r_sim.shape[1] = ' + str(r_sim.shape[1]))
        #     continue
        # if r_ob.shape[1] >= 2300:  # 68
        #     print('skipping because r_ob.shape[1] = ' + str(r_ob.shape[1]))
        #     continue
        # if r_co2_profile_ap.shape[1] < 20:
        #     continue
        # if r_co2_profile.shape[1] < 20:
        #     continue
        # get wavelengths
        w = f.get('SpectralParameters/wavelength')[()]
        # get vars
        vars = []  # list that contains the vars
        for v in var_names:
            # print(v) # uncomment for debugging
            var = f.get(v)[()]
            if var.ndim == 2:  # for 2D variables (e.g. CO2 profile)
                # split up into multiple variables
                for i in range(var.shape[1]):
                    var_i = var[:, i]
                    vars.append(var_i)
                    state_names.append(v + str(i))
            else:
                vars.append(var)
                state_names.append(v)
        # make into numpy array
        vars = np.stack(vars, 1)
        # if vars.shape[1] <= len(var_names):  # 68
        #     print('skipping because vars.shape[1] = ' + str(vars.shape[1]))
        #     continue

        # add zero padding to refs, wls, uncert, and sample_idx
        r_sim_pad = np.zeros((r_sim.shape[0], 2300))
        r_sim_pad[:,:r_sim.shape[1]] = r_sim
        r_sim = r_sim_pad

        r_ob_pad = np.zeros((r_ob.shape[0], 2300))
        r_ob_pad[:,:r_ob.shape[1]] = r_ob
        r_ob = r_ob_pad

        r_uncert_pad = np.zeros((r_uncert.shape[0], 2300))
        r_uncert_pad[:,:r_uncert.shape[1]] = r_uncert
        r_uncert = r_uncert_pad

        r_sample_idx_pad = np.zeros((r_sample_idx.shape[0], 2300))
        r_sample_idx_pad[:,:r_sample_idx.shape[1]] = r_sample_idx
        r_sample_idx = r_sample_idx_pad

        w_pad = np.zeros((w.shape[0], 2300))
        w_pad[:,:w.shape[1]] = w
        w = w_pad


        # append to list
        ref_sims.append(r_sim)
        ref_obs.append(r_ob)
        uncerts.append(r_uncert)
        sample_idxs.append(r_sample_idx)
        num_colors.append(r_color)
        wls.append(w)
        f_vars.append(vars)
        # press_weights.append(r_press_weight)
        # co2_profile_aps.append(r_co2_profile_ap)
        # co2_profiles.append(r_co2_profile)

    if ref_sims == []:
        print('no data for this day')
        return

    
    ref_sims = np.concatenate(ref_sims, 0)
    ref_obs = np.concatenate(ref_obs, 0)
    uncerts = np.concatenate(uncerts, 0)
    sample_idxs = np.concatenate(sample_idxs, 0)
    num_colors = np.concatenate(num_colors, 0)
    wls = np.concatenate(wls, 0)
    states = np.concatenate(f_vars, 0)
    # press_weights = np.concatenate(press_weights, 0)
    # co2_profile_aps = np.concatenate(co2_profile_aps, 0)
    # co2_profiles = np.concatenate(co2_profiles, 0)

        # remove glint land
    if not nadir:
            idx = state_names.index('RetrievalGeometry/retrieval_land_fraction')
            ref_sims = ref_sims[states[:, idx] ==100, :]
            ref_obs = ref_obs[states[:, idx] ==100, :]
            uncerts = uncerts[states[:, idx] ==100, :]
            wls = wls[states[:, idx] ==100, :]
            states = states[states[:, idx] ==100, :]
            num_colors = num_colors[states[:, idx] ==100, :]
            # co2_profile_aps = co2_profile_aps[states[:, idx] ==100, :]
            # co2_profiles = co2_profiles[states[:, idx] ==100, :]

    # remove data with clouds
    idx = state_names.index('PreprocessingResults/cloud_flag_idp')
    ref_sims = ref_sims[states[:, idx] == 3, :]
    ref_obs = ref_obs[states[:, idx] == 3, :]
    uncerts = uncerts[states[:, idx] == 3, :]
    wls = wls[states[:, idx] == 3, :]
    num_colors = num_colors[states[:, idx] == 3, :]
    # co2_profile_aps = co2_profile_aps[states[:, idx] == 3, :]
    # co2_profiles = co2_profiles[states[:, idx] == 3, :]
    states = states[states[:, idx] == 3, :]
    states = np.delete(states, idx, 1)
    state_names.remove('PreprocessingResults/cloud_flag_idp')

    # downsample data to be saved
    if downsample:
        if nadir:
            percent_sample = 0.9
        else:
            percent_sample = 0.5
        # randomly sample the array
        np.random.seed(0)
        idx = np.random.choice(ref_sims.shape[0], int(ref_sims.shape[0]*percent_sample), replace=False)
        ref_sims = ref_sims[idx, :]
        wls = wls[idx, :]
        ref_obs = ref_obs[idx, :]
        uncerts = uncerts[idx, :]
        states = states[idx, :]
        num_colors = num_colors[idx, :]
        # co2_profile_aps = co2_profile_aps[idx, :]
        # co2_profiles = co2_profiles[idx, :]

    print('saving pickle ...')
    save_data = {'ref_sim': ref_sims, 'ref_obs' : ref_obs, 'wl': wls, 'state': states, 'state_var': state_names,
                'num_color': num_colors, 'sample_idxs' : sample_idxs, 'uncerts' : uncerts}
    pickle.dump(save_data, open(save_name, "wb"), protocol=4)
    print('done')

    # save everything as pickle file


    # save everything as pickle file
    # print('o2_band length : ', len(o2_band))
    # if auto_encoder:
    #     save_data = {'o2_no_eof' : o2_band_no_eof, 'o2_with_eof' : o2_band, 'o2_obs' : o2_band_obs,
    #                     'wco2_no_eof' : wco2_band_no_eof, 'wco2_with_eof' : wco2_band, 'wco2_obs' : wco2_band_obs,
    #                     'sco2_no_eof' : sco2_band_no_eof, 'sco2_with_eof' : sco2_band, 'sco2_obs' : sco2_band_obs,
    #                     'wl': wls, 'state': states, 'state_var': state_names}
    #     pickle.dump(save_data, open(save_name, "wb"), protocol=4)
    #     print('done')
    #     # close the files in the glob
    #     f.close()
    # else:
    #     save_data = {'o2_no_eof' : o2_band_no_eof, 'o2_obs' : o2_band_obs,
    #                     'wco2_no_eof' : wco2_band_no_eof, 'wco2_obs' : wco2_band_obs,
    #                         'sco2_no_eof' : sco2_band_no_eof, 'sco2_obs' : sco2_band_obs,
    #             'wl': wls, 'state': states, 'state_var': state_names, 'num_colors': num_colors,
    #             'co2_profile': co2_profile, 'co2_profile_prior' : co2_profile_ap}
    #     pickle.dump(save_data, open(save_name, "wb"), protocol=4)
    #     print('done')
    #     # close the files in the glob
    #     f.close()
            





if __name__ == '__main__':
    years = ['2015']
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    days = ['01','02','03','04','05','06','07','08','09','10',
            '11','12','13','14','15','16','17','18','19','20',
            '21','22','23','24','25','26','27','28','29']
    # days = None
    
    with Pool(2) as p:
        p.map(main, make_pool_map(years,months,days))

        
            
            