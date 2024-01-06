import numpy as np
import pickle
import glob
from tqdm import tqdm

from multiprocessing import Pool


# William Keely william.r.keely [at] jpl.nasa.gov; Steffen Mauceri steffen.mauceri [at] jpl.nasa.gov
# Script that loads wl aligned pickles and concatenates them into a single pickle file for a given year
# and down samples the data to 5M samples

# TODO : add auto encoder test pickle output    

def make_pool_map(years, months, days):
    if days is None:
        pool_map = []
        for year in years:
            for month in months:
                pool_map.append((year, month))
    else:
        pool_map = []
        for year in years:
            for month in months:
                for day in days:
                    pool_map.append((year, month, day))
    
    return pool_map

def main(pool_map, nadir = True, debug = False):
    year = pool_map[0]
    month = pool_map[1]
    if len(pool_map) == 3:
        day = pool_map[2]

    if nadir:
        save_name = './L2DiaND_XCO2_' + str(year) + '_aligned.p'
        files = glob.glob('./Data/'+str(year)+'/L2DiaND_XCO2_'+str(year)+'_'+str(month)+'_'+str(day)+'_*.p')
    else:
        save_name = './L2DiaGL_XCO2_' + str(year) + '_aligned.p'
        files = glob.glob('./Data/'+str(year)+'/L2DiaGL_XCO2_'+str(year)+'_'+str(month)+'_'+str(day)+'_*.p')

    if debug:
        files = files[:100]
    
    ref_sim = []
    ref_obs = []
    wl = []
    state = []
    stat_var = []
    for p in files:
        file = pickle.load(open(p, 'rb'))
        ref_p = file['ref_sim'] #reflectance
        ref_obs_p = file['ref_obs'] #reflectance observations
        wl_p = file['wl'] #wavenumber
        state_p = file['state'] #atmospheric state
        state_var_p = file['state_var'] #name of state variables

        ref_sim.append(ref_p)
        ref_obs.append(ref_obs_p)
        wl.append(wl_p)
        state.append(state_p)
        stat_var.append(state_var_p)
        state_var = stat_var[0]

    ref_sim = np.vstack(ref_sim)
    ref_obs = np.vstack(ref_obs)
    wl = np.vstack(wl)
    state = np.vstack(state)
    print('Shape before downsampling ',ref_sim.shape)

    # randomly subset the array
    np.random.seed(0)
    idx = np.random.choice(ref_sim.shape[0], 5000000, replace=False)
    ref_sim = ref_sim[idx, :]
    ref_obs = ref_obs[idx, :]
    wl = wl[idx, :]
    state = state[idx, :]
    print('Shape after downsampling ',ref_sim.shape)

    # save as pickle
    save_data = {'ref_sim': ref_sim, 'wl': wl, 'ref_obs': ref_obs, 'state': state, 'state_var': state_var}
    pickle.dump(save_data, open(save_name, "wb"), protocol=4)

if __name__ == '__main__':
    years = ['2015'] # only run single year at a time
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    days = ['01','02','03','04','05','06','07','08','09','10',
            '11','12','13','14','15','16','17','18','19','20',
            '21','22','23','24','25','26','27','28','29','30','31']
    pool_map = make_pool_map(years, months, days)
    with Pool(4) as p:
        p.map(main, pool_map)


