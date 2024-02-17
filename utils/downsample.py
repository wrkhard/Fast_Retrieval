import numpy as np
import pickle
import glob
import argparse


def _downsample_year(year_path,save_path, percent_sample = 0.4):
    file = pickle.load(open(year_path, 'rb'))
    ref_sim = file['ref_sim']
    ref_obs = file['ref_obs']
    wl = file['wl']
    state = file['state']
    num_colors = file['num_color']
    sample_idxs = file['sample_idxs']
    uncerts = file['uncerts']
    state_var = file['state_var']
    print('Shape before downsampling ',ref_sim.shape)
    num_samples = ref_sim.shape[0]
    num_samples_to_keep = int(num_samples * percent_sample)
    # randomly subset the array
    np.random.seed(0)
    idx = np.random.choice(num_samples, num_samples_to_keep, replace=False)
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
    pickle.dump(save_data, open(save_path, "wb"), protocol=4)
    print('year saved to ', save_path)

def main(year, mode, year_path, save_path, percent_sample):
    assert mode in ['ND', 'GL']
    year_path = year_path + '/L2Dia'+str(mode)+'_XCO2_'+str(year)+'.p'
    save_path = save_path + '/L2Dia'+str(mode)+'_XCO2_'+str(year)+'_downsampled.p'
    _downsample_year(year_path, save_path, percent_sample)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--mode', type=str, required=False, default="ND")
    parser.add_argument('--year_path', type=str, required=False, default="/scratch-science/algorithm/wkeely/")
    parser.add_argument('--save_path', type=str, required=False, default="/scratch-science/algorithm/wkeely/")
    parser.add_argument('--percent_sample', type=float, default=0.4)
    args = parser.parse_args()
    main(args.year, args.mode, args.year_path, args.save_path, args.percent_sample)
