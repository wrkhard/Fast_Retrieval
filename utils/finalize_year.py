import numpy as np
import pickle
import glob
import argparse


def _generate_month_pickles(mode, year, months,month_path):

    if months is None:
        months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    if month_path is None:
        month_path = '/scratch-science/algorithm/wkeely/Data/'
    
    for month in months:
        ref_sim = []
        ref_obs = []
        wl = []
        state = []
        num_colors = []
        sample_idxs = []
        uncerts = []
        stat_var = []

        files = glob.glob(month_path + 'L2Dia'+str(mode)+'_XCO2_'+str(year)+'_'+str(month)+'_*.p')
        files.sort()
        save_month = '/scratch-science/algorithm/wkeely' + '/L2Dia'+str(mode)+'_XCO2_'+str(year)+'_concat_'+str(month)+'.p'

        for p in files:

            file = pickle.load(open(p, 'rb'))
            ref_p = file['ref_sim'] #reflectance
            ref_obs_p = file['ref_obs'] #reflectance observations
            wl_p = file['wl'] #wavenumber
            state_p = file['state'] #atmospheric state
            num_color_p = file['num_color'] #number of colors
            sample_idxs_p = file['sample_idxs'] #sample indices
            uncert_p = file['uncerts'] #uncertainty in radiance
            state_var_p = file['state_var'] #name of state variables

            ref_sim.append(ref_p)
            ref_obs.append(ref_obs_p)
            wl.append(wl_p)
            state.append(state_p)
            num_colors.append(num_color_p)
            sample_idxs.append(sample_idxs_p)
            uncerts.append(uncert_p)
            stat_var.append(state_var_p)
            state_var = stat_var[0]

        ref_sim = np.vstack(ref_sim)
        ref_obs = np.vstack(ref_obs)
        wl = np.vstack(wl)
        state = np.vstack(state)
        num_colors = np.vstack(num_colors)
        sample_idxs = np.vstack(sample_idxs)
        uncerts = np.vstack(uncerts)
        print('Shape before downsampling ',ref_sim.shape)

        # randomly subset the array
        percent_sample = 0.6
        np.random.seed(0)
        idx = np.random.choice(ref_sim.shape[0], int(percent_sample*ref_sim.shape[0]), replace=False)
        ref_sim = ref_sim[idx, :]
        ref_obs = ref_obs[idx, :]
        wl = wl[idx, :]
        state = state[idx, :]
        num_colors = num_colors[idx, :]
        sample_idxs = sample_idxs[idx, :]
        uncerts = uncerts[idx, :]
        print('Shape after downsampling ',ref_sim.shape)

        # save as pickle
        save_data = {'ref_sim': ref_sim, 'wl': wl, 'ref_obs': ref_obs, 'state': state, 'state_var': state_var, 'num_color': num_colors, 'sample_idxs': sample_idxs, 'uncerts': uncerts}
        pickle.dump(save_data, open(save_month, "wb"), protocol=4)

        print('month saved to ', save_month)


def _combine_months(mode, year, month_path,):
    
    ref_sim = []
    ref_obs = []
    wl = []
    state = []
    num_colors = []
    sample_idxs = []
    uncerts = []
    stat_var = []

    pickles = glob.glob(month_path + '/L2Dia'+str(mode)+'_XCO2_'+str(year)+'_concat_*.p')

    for p in pickles:
        file = pickle.load(open(p, 'rb'))
        ref_p = file['ref_sim']
        ref_obs_p = file['ref_obs']
        wl_p = file['wl']
        state_p = file['state']
        num_color_p = file['num_color']
        sample_idxs_p = file['sample_idxs']
        uncert_p = file['uncerts']
        state_var_p = file['state_var']
        print('pickle file : ', p)
        # print('number of soundings in pickle',len(ref_sim))

        ref_sim.append(ref_p)
        ref_obs.append(ref_obs_p)
        wl.append(wl_p)
        state.append(state_p)
        num_colors.append(num_color_p)
        sample_idxs.append(sample_idxs_p)
        uncerts.append(uncert_p)
        stat_var.append(state_var_p)
        state_var = stat_var[0]

    ref_sim = np.vstack(ref_sim)
    ref_obs = np.vstack(ref_obs)
    wl = np.vstack(wl)
    state = np.vstack(state)
    num_colors = np.vstack(num_colors)
    sample_idxs = np.vstack(sample_idxs)
    uncerts = np.vstack(uncerts)

    # down sample the data
    print('Shape before downsampling ',ref_sim.shape)
    percent_sample = 0.6
    np.random.seed(0)
    idx = np.random.choice(ref_sim.shape[0], int(percent_sample*ref_sim.shape[0]), replace=False)
    ref_sim = ref_sim[idx, :]
    ref_obs = ref_obs[idx, :]
    wl = wl[idx, :]
    state = state[idx, :]
    num_colors = num_colors[idx, :]
    sample_idxs = sample_idxs[idx, :]
    uncerts = uncerts[idx, :]
    print('Shape after downsampling ',ref_sim.shape)

    # save as pickle
    save_data = {'ref_sim': ref_sim, 'wl': wl, 'ref_obs': ref_obs, 'state': state, 'state_var': state_var, 'num_color': num_colors, 'sample_idxs': sample_idxs, 'uncerts': uncerts}
    return save_data

def main(mode,year, month_path, save_path):
    assert mode in ['ND', 'GL'], "mode must be ND or GL"
    _generate_month_pickles(mode, year, None, None)
    data = _combine_months(mode, year, month_path)
    pickle.dump(data, open(save_path + 'L2Dia'+str(mode)+'_XCO2_'+str(year)+'.p', "wb"), protocol=4)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="ND or GL", default="ND")
    parser.add_argument("--year", type=str, help="year", default = 2015)
    parser.add_argument("--month_path", type=str, help="path to the months", default="/scratch-science/algorithm/wkeely/")
    parser.add_argument("--save_path", type=str, help="path to save the combined year", default="/scratch-science/algorithm/wkeely/")
    args = parser.parse_args()

    main(args.mode, args.year, args.month_path, args.save_path)

 
