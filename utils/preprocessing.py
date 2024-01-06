import numpy as np
import h5py as h5
import pickle
import glob
from multiprocessing import Pool

def make_pool_map(years, months, days):
    year_month = []
    if days is not None:
        for year in years:
            for month in months:
                for day in days:
                    year_month.append((year,month,day))
    else:
        for year in years:
            for month in months:
                year_month.append((year,month))

    return year_month


def main(pool_map,nadir = False,downsample = True,remove_eofs =True,debug = True, auto_encoder_test = False):
    
    year = pool_map[0]
    month = pool_map[1]
    day = pool_map[2]
    
    
    if nadir:
        save_name = './L2DiaND_XCO2_' + year + '_'+month+'_'+day+'.p'
        files = glob.glob('/data/oco2/ops/product/Ops_B11006_r02/' + year + '/'+ month+'/'+day+'/L2Dia/oco2_L2DiaND*.h5')

        if debug:
            files = files[::100] # for debugging. Comment out later for full data.
        
        ref_sim = 'SpectralParameters/modeled_radiance'
        ref_ob = 'SpectralParameters/measured_radiance'
        ob_uncert = 'SpectralParameters/measured_radiance_uncert'
        sample_idx = 'SpectralParameters/sample_index'
        num_color = 'SpectralParameters/num_colors_per_band'


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
                        'PreprocessingResults/cloud_flag_idp']


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
        for file in files:
            print(file)
            f = h5.File(file, 'r')
            # get radiance
            r_sim = f.get(ref_sim)[()]
            r_ob = f.get(ref_ob)[()]
            r_uncert = f.get(ob_uncert)[()]
            r_sample_idx = f.get(sample_idx)[()]
            r_color = f.get(num_color)[()]
            if len(r_sim) < 10:
                print('skipping because len(r_sim) = ' + str(len(r_sim)))
                continue
            if len(r_ob) < 10:
                print('skipping because len(r_ob) = ' + str(len(r_ob)))
                continue
            if r_sim.shape[1] >= 2300:  # 68
                print('skipping because r_sim.shape[1] = ' + str(r_sim.shape[1]))
                continue
            if r_ob.shape[1] >= 2300:  # 68
                print('skipping because r_ob.shape[1] = ' + str(r_ob.shape[1]))
                continue
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

            # append to list
            ref_sims.append(r_sim)
            ref_obs.append(r_ob)
            uncerts.append(r_uncert)
            sample_idxs.append(r_sample_idx)
            num_colors.append(r_color)
            wls.append(w)
            f_vars.append(vars)

        ref_sims = np.concatenate(ref_sims, 0)
        ref_obs = np.concatenate(ref_obs, 0)
        uncerts = np.concatenate(uncerts, 0)
        sample_idxs = np.concatenate(sample_idxs, 0)
        num_colors = np.concatenate(num_colors, 0)
        wls = np.concatenate(wls, 0)
        states = np.concatenate(f_vars, 0)

        if remove_eofs:
            print('Removing EOFs ...')

            # EOF path
            eof_path = './EOF/OCO2_qtsb11_eofs_00000-40000_oceanG_alt2_falt1_L2.h5'
            eof_file = h5.File(eof_path, 'r')

            # EOFs for O2A band for each footprint
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
            idx = state_names.index('RetrievalHeader/sounding_id')
            sid = states[:, idx]
            footprints = sid % 10
            footprints = footprints.astype(int)

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

            ref_sims_no_eof = []

            # replace sample indices with -2147483647 with np.nan
            # sample_idxs[sample_idxs == -2147483647] = np.nan

            for idx, fp in enumerate(footprints):
                print(idx, fp)
                print(ref_sims[idx].shape)
                print(uncerts[idx].shape)
                print(sample_idxs[idx].shape)
                print(num_colors[idx].shape)
                print(eof_1_o2[fp].shape)

                padding = np.zeros(np.sum(ref_sims[idx]==-9.9999900e+05))
                padding.fill(-9.9999900e+10) # padding will return to e+05 after subtracting off eof
                #print(padding.shape)
                # O2A band
                x = eof_1_o2[fp,sample_idxs[idx,:num_colors[idx,0]]] * uncerts[idx,:num_colors[idx,0]]
                # if idx == 0:
                #     plt.plot(x),plt.show()
                #     print(x.shape)
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
                # if idx == 0:
                #     plt.plot(x),plt.show()
                #     print(x.shape)
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
                # if idx == 0:
                #     plt.plot(x),plt.show()
                #     print(x.shape)
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
                scaled_eof = np.concatenate((term1,term2,term3,padding),axis=0)
                ref_sims_no_eof.append(ref_sims[idx,:] - scaled_eof)


            # same shape as ref_sims
            ref_sims_no_eof = np.stack(ref_sims_no_eof,0)


            # remove data with clouds                                                                                          
            idx = state_names.index('PreprocessingResults/cloud_flag_idp')
            ref_sims = ref_sims[states[:, idx] == 3, :]
            ref_sims_no_eof = ref_sims_no_eof[states[:, idx] == 3, :]
            ref_obs = ref_obs[states[:, idx] == 3, :]
            uncerts = uncerts[states[:, idx] == 3, :]
            wls = wls[states[:, idx] == 3, :]
            states = states[states[:, idx] == 3, :]
            states = np.delete(states, idx, 1)
            state_names.remove('PreprocessingResults/cloud_flag_idp')

            # remove very long wavelengths with a lot of missing data                                                          
            ref_sims = ref_sims[:, :2100]
            ref_sims_no_eof = ref_sims_no_eof[:, :2100]
            wls = wls[:, :2100]
            ref_obs = ref_obs[:, :2100]
            uncerts = uncerts[:, :2100]

            # downsample data to be saved                                                                                      
            if downsample:
                percent_sample = 0.4
                # randomly sample the array                                                                                    
                np.random.seed(0)
                idx = np.random.choice(ref_sims.shape[0], int(ref_sims.shape[0]*percent_sample), replace=False)
                ref_sims = ref_sims[idx, :]
                wls = wls[idx, :]
                ref_obs = ref_obs[idx, :]
                ref_sims_no_eof = ref_sims_no_eof[idx, :]
                uncerts = uncerts[idx, :]
                states = states[idx, :]

            # save everything as pickle file
            if auto_encoder_test:
                save_data = {'ref_sim_w_eof' : ref_sims,'ref_sim': ref_sims_no_eof, 'ref_obs' : ref_obs, 'wl': wls, 'state': states, 'state_var': state_names}
                pickle.dump(save_data, open(save_name, "wb"), protocol=4)
                print('done')
            else:
                save_data = {'ref_sim': ref_sims_no_eof, 'ref_obs' : ref_obs, 'wl': wls, 'state': states, 'state_var': state_names}
                pickle.dump(save_data, open(save_name, "wb"), protocol=4)
                print('done')
        else:

            # remove data with clouds                                                                                          
            idx = state_names.index('PreprocessingResults/cloud_flag_idp')
            ref_sims = ref_sims[states[:, idx] == 3, :]
            ref_sims_no_eof = ref_sims_no_eof[states[:, idx] == 3, :]
            ref_obs = ref_obs[states[:, idx] == 3, :]
            uncerts = uncerts[states[:, idx] == 3, :]
            wls = wls[states[:, idx] == 3, :]
            states = states[states[:, idx] == 3, :]
            states = np.delete(states, idx, 1)
            state_names.remove('PreprocessingResults/cloud_flag_idp')

            # remove very long wavelengths with a lot of missing data                                                          
            ref_sims = ref_sims[:, :2100]
            ref_sims_no_eof = ref_sims_no_eof[:, :2100]
            wls = wls[:, :2100]
            ref_obs = ref_obs[:, :2100]
            uncerts = uncerts[:, :2100]

            # downsample data to be saved                                                                                      
            if downsample:
                percent_sample = 0.4
                # randomly sample the array                                                                                    
                np.random.seed(0)
                idx = np.random.choice(ref_sims.shape[0], int(ref_sims.shape[0]*percent_sample), replace=False)
                ref_sims = ref_sims[idx, :]
                wls = wls[idx, :]
                ref_obs = ref_obs[idx, :]
                ref_sims_no_eof = ref_sims_no_eof[idx, :]
                uncerts = uncerts[idx, :]
                states = states[idx, :]

            # save everything as pickle file
            save_data = {'ref_sim': ref_sims, 'ref_obs' : ref_obs, 'wl': wls, 'state': states, 'state_var': state_names}
            pickle.dump(save_data, open(save_name, "wb"), protocol=4)
            print('done')


    else:
      
        save_name = './L2DiaGL_XCO2_' + year + '_'+month+'_'+day+'.p'
        files = glob.glob('/data/oco2/ops/product/Ops_B11006_r02/' + year + '/'+ month+'/'+day+'/*/L2Dia/oco2_L2DiaGL*.h5')

        if debug:
            files = files[::100]
        
        ref_sim = 'SpectralParameters/modeled_radiance'
        ref_ob = 'SpectralParameters/measured_radiance'
        ob_uncert = 'SpectralParameters/measured_radiance_uncert'
        sample_idx = 'SpectralParameters/sample_index'
        num_color = 'SpectralParameters/num_colors_per_band'


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
                        'PreprocessingResults/cloud_flag_idp']
        
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

        for file in files:
            print(file)
            f = h5.File(file, 'r')
            # get radiance
            r_sim = f.get(ref_sim)[()]
            r_ob = f.get(ref_ob)[()]
            r_uncert = f.get(ob_uncert)[()]
            r_sample_idx = f.get(sample_idx)[()]
            r_color = f.get(num_color)[()]
            if len(r_sim) < 10:
                print('skipping because len(r_sim) = ' + str(len(r_sim)))
                continue
            if len(r_ob) < 10:
                print('skipping because len(r_ob) = ' + str(len(r_ob)))
                continue
            if r_sim.shape[1] >= 2300:
                print('skipping because r_sim.shape[1] = ' + str(r_sim.shape[1]))
                continue
            if r_ob.shape[1] >= 2300:
                print('skipping because r_ob.shape[1] = ' + str(r_ob.shape[1]))
                continue
            # get wavelengths
            w = f.get('SpectralParameters/wavelength')[()]
            # get vars
            vars = []
            for v in var_names:
                # print(v) # uncomment for debugging
                var = f.get(v)[()]
                if var.ndim == 2:
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
            # if vars.shape[1] <= len(var_names):
            #     print('skipping because vars.shape[1] = ' + str(vars.shape[1]))
            #     continue

            # append to list
            ref_sims.append(r_sim)
            ref_obs.append(r_ob)
            uncerts.append(r_uncert)
            sample_idxs.append(r_sample_idx)
            num_colors.append(r_color)
            wls.append(w)
            f_vars.append(vars)

        ref_sims = np.concatenate(ref_sims, 0)
        ref_obs = np.concatenate(ref_obs, 0)
        uncerts = np.concatenate(uncerts, 0)
        sample_idxs = np.concatenate(sample_idxs, 0)
        num_colors = np.concatenate(num_colors, 0)
        wls = np.concatenate(wls, 0)
        states = np.concatenate(f_vars, 0)

        # remove land glint
        idx = state_names.index('RetrievalGeometry/retrieval_land_fraction')
        ref_sims = ref_sims[states[:, idx] ==100, :]
        ref_obs = ref_obs[states[:, idx] ==100, :]
        uncerts = uncerts[states[:, idx] ==100, :]
        wls = wls[states[:, idx] ==100, :]
        states = states[states[:, idx] ==100, :]
        states = np.delete(states, idx, 1)
        


        if remove_eofs:
            print('Removing EOFs ...')

            # EOF path
            eof_path = './EOF/OCO2_qtsb11_eofs_00000-40000_oceanG_alt2_falt1_L2.h5'
            eof_file = h5.File(eof_path, 'r')

            # EOFs for O2A band for each footprint
            eof_1_o2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Ocean/EOF_1_waveform_1'][:]
            eof_2_o2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Ocean/EOF_2_waveform_1'][:]
            eof_3_o2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Ocean/EOF_3_waveform_1'][:]

            # EOFs for WCO2 band for each footprint
            eof_1_wco2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Ocean/EOF_1_waveform_2'][:]
            eof_2_wco2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Ocean/EOF_2_waveform_2'][:]
            eof_3_wco2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Ocean/EOF_3_waveform_2'][:]

            # EOFs for SCO2 band for each footprint
            eof_1_sco2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Ocean/EOF_1_waveform_3'][:]
            eof_2_sco2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Ocean/EOF_2_waveform_3'][:]
            eof_3_sco2 = eof_file['Instrument/EmpiricalOrthogonalFunction/Ocean/EOF_3_waveform_3'][:]

            # get the footprint which is the last digit of the sounding id
            idx = state_names.index('RetrievalHeader/sounding_id')
            sid = states[:, idx]
            footprints = sid % 10
            footprints = footprints.astype(int)

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

            ref_sims_no_eof = []

            # replace sample indices with -2147483647 with np.nan
            # sample_idxs[sample_idxs == -2147483647] = np.nan

            for idx, fp in enumerate(footprints):
                print(idx, fp)
                print(ref_sims[idx].shape)
                print(uncerts[idx].shape)
                print(sample_idxs[idx].shape)
                print(num_colors[idx].shape)
                print(eof_1_o2[fp].shape)

                padding = np.zeros(np.sum(ref_sims[idx]==-9.9999900e+05))
                padding.fill(-9.9999900e+10)
                #print(padding.shape)
                # O2A band
                x = eof_1_o2[fp,sample_idxs[idx,:num_colors[idx,0]]] * uncerts[idx,:num_colors[idx,0]]
                # if idx == 0:
                #     plt.plot(x),plt.show()
                #     print(x.shape)
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
                # if idx == 0:
                #     plt.plot(x),plt.show()
                #     print(x.shape)
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
                # if idx == 0:
                #     plt.plot(x),plt.show()
                #     print(x.shape)
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
                scaled_eof = np.concatenate((term1,term2,term3,padding),axis=0)
                ref_sims_no_eof.append(ref_sims[idx,:] - scaled_eof)


            # same shape as ref_sims
            ref_sims_no_eof = np.stack(ref_sims_no_eof,0)


            # remove data with clouds
            idx = state_names.index('PreprocessingResults/cloud_flag_idp')
            ref_sims = ref_sims[states[:, idx] == 3, :]
            ref_sims_no_eof = ref_sims_no_eof[states[:, idx] == 3, :]
            ref_obs = ref_obs[states[:, idx] == 3, :]
            uncerts = uncerts[states[:, idx] == 3, :]
            wls = wls[states[:, idx] == 3, :]
            states = states[states[:, idx] == 3, :]
            states = np.delete(states, idx, 1)
            state_names.remove('PreprocessingResults/cloud_flag_idp')

            # remove very long wavelengths with a lot of missing data
            ref_sims = ref_sims[:, :2100]
            ref_sims_no_eof = ref_sims_no_eof[:, :2100]
            wls = wls[:, :2100]
            ref_obs = ref_obs[:, :2100]
            uncerts = uncerts[:, :2100]

            # downsample data to be saved
            if downsample:
                percent_sample = 0.4
                # randomly sample the array
                np.random.seed(0)
                idx = np.random.choice(ref_sims.shape[0], int(ref_sims.shape[0]*percent_sample), replace=False)
                ref_sims = ref_sims[idx, :]
                wls = wls[idx, :]
                ref_obs = ref_obs[idx, :]
                ref_sims_no_eof = ref_sims_no_eof[idx, :]
                uncerts = uncerts[idx, :]
                states = states[idx, :]

            # save everything as pickle file
            if auto_encoder_test:
                save_data = {'ref_sim_w_eof' : ref_sims,'ref_sim': ref_sims_no_eof, 'ref_obs' : ref_obs, 'wl': wls, 'state': states, 'state_var': state_names}
                pickle.dump(save_data, open(save_name, "wb"), protocol=4)
                print('done')
            else:
                save_data = {'ref_sim': ref_sims_no_eof, 'ref_obs' : ref_obs, 'wl': wls, 'state': states, 'state_var': state_names}
                pickle.dump(save_data, open(save_name, "wb"), protocol=4)
                print('done')
        else:

            # remove data with clouds
            idx = state_names.index('PreprocessingResults/cloud_flag_idp')
            ref_sims = ref_sims[states[:, idx] == 3, :]
            ref_sims_no_eof = ref_sims_no_eof[states[:, idx] == 3, :]
            ref_obs = ref_obs[states[:, idx] == 3, :]
            uncerts = uncerts[states[:, idx] == 3, :]
            wls = wls[states[:, idx] == 3, :]
            states = states[states[:, idx] == 3, :]
            states = np.delete(states, idx, 1)
            state_names.remove('PreprocessingResults/cloud_flag_idp')

            # remove very long wavelengths with a lot of missing data
            ref_sims = ref_sims[:, :2100]
            ref_sims_no_eof = ref_sims_no_eof[:, :2100]
            wls = wls[:, :2100]
            ref_obs = ref_obs[:, :2100]
            uncerts = uncerts[:, :2100]

            # downsample data to be saved
            if downsample:
                percent_sample = 0.4
                # randomly sample the array
                np.random.seed(0)
                idx = np.random.choice(ref_sims.shape[0], int(ref_sims.shape[0]*percent_sample), replace=False)
                ref_sims = ref_sims[idx, :]
                wls = wls[idx, :]
                ref_obs = ref_obs[idx, :]
                ref_sims_no_eof = ref_sims_no_eof[idx, :]
                uncerts = uncerts[idx, :]
                states = states[idx, :]

            # save everything as pickle file
            save_data = {'ref_sim': ref_sims, 'ref_obs' : ref_obs, 'wl': wls, 'state': states, 'state_var': state_names}
            pickle.dump(save_data, open(save_name, "wb"), protocol=4)
            print('done')



                            
 # MAKE CHANGES HERE: run one year at a time. Finalize soundings for that year and then clear scratch-science.

if __name__ == '__main__':
    years = ['2015']
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    # days_odd = ['01','03','05','07','09','11','13','15','17','19','21','23','25','27','29'] # use for glint
    days_all = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15',
                '16','17','18','19','20','21','22','23','24','25','26','27','28','29']
    with Pool(2) as p:
        pool_map = make_pool_map(years, months, days_all)
        # pool_map = make_pool_map(years, months, )
        p.map(main, pool_map)
