import numpy as np
import pickle
import glob
from tqdm import tqdm

from multiprocessing import Pool

# William Keely william.r.keely [at] jpl.nasa.gov; Steffen Mauceri steffen.mauceri [at] jpl.nasa.gov
# Script to align the wavelengths of the reflectance simulations and reflectance observations

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


def main(pool_map, t = 0.0002, nadir = True, debug = True):
    year = pool_map[0]
    month = pool_map[1]
    if len(pool_map) == 3:
        day = pool_map[2]


    if nadir:
        save_name = './L2DiaND_XCO2_' + str(year) + '_' + str(month) + '_'+ str(day) +'.p'
        files = glob.glob('./Data/'+str(year)+'/L2DiaND_XCO2_'+str(year)+'_'+str(month)+'_'+str(day)+'_*.p')
    else:
        save_name = './L2DiaGL_XCO2_' + str(year) + '_' + str(month) + '_' + str(day) + '.p'
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

    ref_sim = np.vstack(ref_sim)
    ref_obs = np.vstack(ref_obs)
    wl = np.vstack(wl)
    state = np.vstack(state)
    stat_var = stat_var[0][:4]

    # # remove wavelengths with gaps
    wl_diff = []
    for i in tqdm(range(len(wl))):
        diff = np.diff(wl[i, :])

        wl_diff.append(np.max(diff[diff <= 0.4]))
    wl_diff = np.array(wl_diff)

    # t is the threshold for removing wavelengths with gaps 
    ref = ref[wl_diff < t,:]
    state = state[wl_diff < t,:]
    wl = wl[wl_diff < t,:]

    # find common wavelengths grid
    wl_med = np.median(wl[::5,:],0)

    # interpolate wavelengths on common wavelength grid
    ref_meds_sim = []
    ref_meds_obs =[]
    for i in tqdm(range(len(wl))):
        ref_s = np.interp(wl_med, wl[i,:],ref[i,:], left=0, right=0)
        ref_meds_sim.append(ref_s)
        ref_o = np.interp(wl_med, wl[i,:],ref_obs[i,:], left=0, right=0)
        ref_meds_obs.append(ref_o)
        
    ref_med_sim = np.vstack(ref_meds_sim)
    ref_med_obs = np.vstack(ref_meds_obs)

    # remove columns with zeros
    ref_med_sim = ref_med_sim[:,1:-10]
    ref_med_obs = ref_med_obs[:,1:-10]
    wl_med = wl_med[1:-10]

    # remove rows with zeros
    state = state[np.min(ref_med_sim,1)>0,:]
    ref_med_sim = ref_med_sim[np.min(ref_med_sim,1)>0,:]
    ref_med_obs = ref_med_obs[np.min(ref_med_obs,1)>0,:]

    # change units to something more managable
    # state[:,0] = state[:,0]*10**6

    print(str(len(ref_med_sim)) + ' soundings')

    #save everything as pickle file
    save_data = {'ref_sim': ref_med_sim,'ref_obs' : ref_med_obs , 'wl':wl_med, 'state': state, 'state_var': stat_var}
    pickle.dump( save_data, open( save_name, "wb" ), protocol = 4)
    print('done')



if __name__ == '__main__':
    years = ['2015'] # just run one year at a time
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    days = ['01','02','03','04','05','06','07','08','09','10',
            '11','12','13','14','15','16','17','18','19','20',
            '21','22','23','24','25','26','27','28','29','30','31']
    pool_map = make_pool_map(years, months, days)
    with Pool(1) as p:
        p.map(main, pool_map)



    