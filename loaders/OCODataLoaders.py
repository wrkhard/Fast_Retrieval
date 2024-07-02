"""
#
#
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Contact: william.r.keely<at>jpl.nasa.gov



Dataset and DataLoader classes for OCO-2 Xgas retrieval.

state vars:
index : 0   RetrievalResults/surface_pressure_apriori_fph
index : 1   RetrievalResults/wind_speed_apriori
index : 2   RetrievalResults/eof_1_scale_weak_co2
index : 3   RetrievalResults/eof_1_scale_strong_co2
index : 4   RetrievalResults/eof_1_scale_o2
index : 5   RetrievalResults/eof_2_scale_weak_co2
index : 6   RetrievalResults/eof_2_scale_strong_co2
index : 7   RetrievalResults/eof_2_scale_o2
index : 8   RetrievalResults/eof_3_scale_weak_co2
index : 9   RetrievalResults/eof_3_scale_strong_co2
index : 10   RetrievalResults/eof_3_scale_o2
index : 11   RetrievalGeometry/retrieval_solar_zenith
index : 12   RetrievalGeometry/retrieval_zenith
index : 13   RetrievalGeometry/retrieval_solar_azimuth
index : 14   RetrievalGeometry/retrieval_azimuth
index : 15   PreprocessingResults/surface_pressure_apriori_abp
index : 16   PreprocessingResults/dispersion_multiplier_abp
index : 17   RetrievalHeader/sounding_id
index : 18   RetrievalGeometry/retrieval_longitude
index : 19   RetrievalGeometry/retrieval_latitude
index : 20   RetrievalResults/xco2

"""




import numpy as np
import pickle

import torch
from torch import Tensor
from torch.utils.data import Dataset
import lightning.pytorch as pl
from torch.utils.data import DataLoader

class RetrievalDataSet(Dataset):
    '''OCO-2 DataSet for Xgas retrieval'''
    def __init__(self, files, train=True, normalize=True, use_convolutions=False, transform=None, normalize_file='normalize.pkl'):
        self.files = files
        self.train = train
        self.normalize = normalize
        self.transform = transform
        self.normalize_file = normalize_file
        self.use_convolutions = use_convolutions
        
        # Load data based on the file paths
        self.X, self.y = self.load_data()

    def load_data(self):
        sco2 = []
        wco2 = []
        o2 = []
        psurf_prior = []
        geometry = []
        y = []

        
        for file_path in self.files:
            try:
                with open(file_path, 'rb') as file:
                    file_data = pickle.load(file)
                    
                    if self.train:
                        print("shape of sco2 : ", file_data['sco2_band_no_eof'].shape)
                        print("shape of wco2 : ", file_data['wco2_band_no_eof'].shape)
                        print("shape of o2 : ", file_data['o2_band_no_eof'].shape)
                        sco2.append(file_data['sco2_band_no_eof'])
                        wco2.append(file_data['wco2_band_no_eof'])
                        o2.append(file_data['o2_band_no_eof'])
                    else:
                        sco2.append(file_data['sco2_band_obs'])
                        wco2.append(file_data['wco2_band_obs'])
                        o2.append(file_data['o2_band_obs'])
                    psurf = file_data['states'][:,0]
                    geom = file_data['states'][:,11:15]
                    xco2 = file_data['states'][:, 20]
                    psurf = np.array(psurf)
                    geom = np.array(geom)
                    xco2 = np.array(xco2)
                    psurf = psurf.reshape(-1, 1)
                    geom = geom.reshape(-1, 4)
                    xco2 = xco2.reshape(-1, 1)
                    # check that psurf, geom and xco2 are of shape (n_samples, 1)

                    psurf_prior.append(psurf)
                    geometry.append(geom)
                    y.append(xco2)

                    print('xco2 shape : ', xco2.shape)
                    print('xco2 values : ', xco2)

                    
            except (FileNotFoundError, KeyError, pickle.UnpicklingError) as e:
                print(f"Error loading file {file_path}: {e}")


        
        if self.normalize:
            if not self.train:
                # make everything a torch tensor
                sco2 = np.vstack(sco2)
                wco2 = np.vstack(wco2)
                o2 = np.vstack(o2)
                psurf_prior = np.vstack(psurf_prior)
                geometry = np.vstack(geometry)
                y = np.vstack(y)

                sco2 = torch.tensor(sco2, dtype=torch.float32)
                wco2 = torch.tensor(wco2, dtype=torch.float32)
                o2 = torch.tensor(o2, dtype=torch.float32)
                psurf_prior = torch.tensor(psurf_prior, dtype=torch.float32)
                geometry = torch.tensor(geometry, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)

                with open(self.normalize_file, 'rb') as file:
                    norm_params = pickle.load(file)
                sco2 = (sco2 - norm_params['sco2_mean']) / norm_params['sco2_std']
                wco2 = (wco2 - norm_params['wco2_mean']) / norm_params['wco2_std']
                o2 = (o2 - norm_params['o2_mean']) / norm_params['o2_std']
                psurf_prior = (psurf_prior - norm_params['psurf_prior_mean']) / norm_params['psurf_prior_std']
                geometry = (geometry - norm_params['geometry_mean']) / norm_params['geometry_std']
                y = (y - norm_params['targets_mean']) / norm_params['targets_std']
                # squeez the data
                if len(sco2.shape) == 3:
                    sco2 = sco2.squeeze(1)
                    wco2 = wco2.squeeze(1)
                    o2 = o2.squeeze(1)
                    psurf_prior = psurf_prior.squeeze(1)
                    geometry = geometry.squeeze(1)
                    y = y.squeeze(1)
                # remove nans
                mask = torch.isnan(sco2).any(axis=1) | torch.isnan(wco2).any(axis=1) | torch.isnan(o2).any(axis=1) | torch.isnan(y).any(axis=1) | torch.isnan(psurf_prior).any(axis=1) | torch.isnan(geometry).any(axis=1)
                sco2 = sco2[~mask]
                wco2 = wco2[~mask]
                o2 = o2[~mask]
                psurf_prior = psurf_prior[~mask]
                geometry = geometry[~mask]
                y = y[~mask]
                # concat sco2, wco2, and o2
                X = torch.cat([o2, wco2, sco2, psurf_prior, geometry], axis=1)

                # return the last 1000 samples for testing


            else:
                sco2 = np.vstack(sco2)
                wco2 = np.vstack(wco2)
                o2 = np.vstack(o2)
                psurf_prior = np.vstack(psurf_prior)
                geometry = np.vstack(geometry)
                y = np.vstack(y)

                print("shape of sco2 : ", sco2.shape)
                print("shape of wco2 : ", wco2.shape)
                print("shape of o2 : ", o2.shape)
                # make everything a torch tensor
                sco2 = torch.tensor(sco2, dtype=torch.float32)
                wco2 = torch.tensor(wco2, dtype=torch.float32)
                o2 = torch.tensor(o2, dtype=torch.float32)
                psurf_prior = torch.tensor(psurf_prior, dtype=torch.float32)
                geometry = torch.tensor(geometry, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                # if any shape length is 3 (i.e. [n_samples, 1, n_features]), then squeeze
                if len(sco2.shape) == 3:
                    sco2 = sco2.squeeze(1)
                    wco2 = wco2.squeeze(1)
                    o2 = o2.squeeze(1)
                    psurf_prior = psurf_prior.squeeze(1)
                    geometry = geometry.squeeze(1)
                    y = y.squeeze(1)
                # remove nans
                mask = torch.isnan(sco2).any(axis=1) | torch.isnan(wco2).any(axis=1) | torch.isnan(o2).any(axis=1) | torch.isnan(y).any(axis=1) | torch.isnan(psurf_prior).any(axis=1) | torch.isnan(geometry).any(axis=1)
                sco2 = sco2[~mask]
                wco2 = wco2[~mask]
                o2 = o2[~mask]
                psurf_prior = psurf_prior[~mask]
                geometry = geometry[~mask]
                y = y[~mask]
                # calculate the mean and std for normalization
                sco2_mean = torch.mean(sco2)
                sco2_std = torch.std(sco2)
                wco2_mean = torch.mean(wco2)
                wco2_std = torch.std(wco2)
                o2_mean = torch.mean(o2)
                o2_std = torch.std(o2)
                psurf_prior_mean = torch.mean(psurf_prior)
                psurf_prior_std = torch.std(psurf_prior)
                geometry_mean = torch.mean(geometry)
                geometry_std = torch.std(geometry)
                targets_mean = torch.mean(y)
                targets_std = torch.std(y)
                sco2 = (sco2 - sco2_mean) / sco2_std
                wco2 = (wco2 - wco2_mean) / wco2_std
                o2 = (o2 - o2_mean) / o2_std
                psurf_prior = (psurf_prior - psurf_prior_mean) / psurf_prior_std
                geometry = (geometry - geometry_mean) / geometry_std
                y = (y - targets_mean) / targets_std
                # concat sco2, wco2, and o2
                X = torch.cat([o2, wco2, sco2, psurf_prior, geometry], axis=1)

                with open(self.normalize_file, 'wb') as file:
                    pickle.dump({'sco2_mean': sco2_mean, 'sco2_std': sco2_std,
                                 'wco2_mean': wco2_mean, 'wco2_std': wco2_std,
                                 'o2_mean': o2_mean, 'o2_std': o2_std,
                                    'psurf_prior_mean': psurf_prior_mean, 'psurf_prior_std': psurf_prior_std,
                                    'geometry_mean': geometry_mean, 'geometry_std': geometry_std,
                                 'targets_mean': targets_mean, 'targets_std': targets_std}, file)
                                   


        print(X.shape)
        assert X.shape[0] == y.shape[0], "Data lengths do not match"
        # y = y.unsqueeze(-1)
        # if use_convolutions is True, reshape the data to [1, number of samples, number of features]
        if self.use_convolutions:
            X = X.unsqueeze(0)
            y = y.unsqueeze(0)


    
        return X, y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> Tensor:
        X = self.X[idx]
        y = self.y[idx]
        
        sample = {'input': X, 'target': y}
        return sample
    

class SimDiffDataSet(Dataset):
    '''OCO-2 DataSet for making retrieval robust to sim - obs differences'''
    def __init__(self, band, files, train=True, normalize=True, use_convolutions=False, train_on_sim = False, normalize_file='sim_diff_normalize.pkl', scaled_eof=False):
        """Args:
            band: The band to use X and y. Options are 'sco2', 'wco2', 'o2', 'all'
            files: The list of file paths
            train: If True, use the training data
            normalize: If True, normalize the data
            train_on_sim: If True, train on the simulated data
            normalize_file: The file to save the normalization parameters
            scaled_eof: If True, use scaled the EOFs
            use_convolutions: If True, reshape the data to [1, number of samples, number of features]
        """
        self.band = band
        self.files = files
        self.train = train
        self.normalize = normalize
        self.train_on_sim = train_on_sim
        self.normalize_file = normalize_file
        self.use_convolutions = use_convolutions
        self.scaled_eof = scaled_eof
        
        # Load data based on the file paths
        self.X, self.y = self.load_data()

    def load_data(self):
        sco2 = []
        wco2 = []
        o2 = []
        y_sco2 = []
        y_wco2 = []
        y_o2 = []
        
        for file_path in self.files:
            try:
                with open(file_path, 'rb') as file:
                    file_data = pickle.load(file)
                    

                    sco2.append(file_data['sco2_band_obs'])
                    wco2.append(file_data['wco2_band_obs'])
                    o2.append(file_data['o2_band_obs'])
                    sco2 = np.squeeze(sco2)
                    wco2 = np.squeeze(wco2)
                    o2 = np.squeeze(o2)
                    sco2 = sco2 * 1E-19
                    wco2 = wco2 * 1E-19
                    o2 = o2 * 1E-19

                   

                    if self.scaled_eof:
                        y_sco2.append(file_data['sco2_band'])
                        y_wco2.append(file_data['wco2_band'])
                        y_o2.append(file_data['o2_band'])
                        y_sco2 = np.squeeze(y_sco2)
                        y_wco2 = np.squeeze(y_wco2)
                        y_o2 = np.squeeze(y_o2)
                        y_sco2 = y_sco2 * 1E-19
                        y_wco2 = y_wco2 * 1E-19
                        y_o2 = y_o2 * 1E-19
                        self.normalize_file = 'scaled_eof_sim_diff_normalize.pkl'
                    else:
                        y_sco2.append(file_data['sco2_band_no_eof'])
                        y_wco2.append(file_data['wco2_band_no_eof'])
                        y_o2.append(file_data['o2_band_no_eof'])
                        y_sco2 = np.squeeze(y_sco2)
                        y_wco2 = np.squeeze(y_wco2)
                        y_o2 = np.squeeze(y_o2)
                        y_sco2 = y_sco2 * 1E-19
                        y_wco2 = y_wco2 * 1E-19
                        y_o2 = y_o2 * 1E-19




                    
            
            except (FileNotFoundError, KeyError, pickle.UnpicklingError) as e:
                print(f"Error loading file {file_path}: {e}")




        if not self.train:

            # make everything a torch tensor
            sco2 = torch.tensor(sco2, dtype=torch.float32)
            wco2 = torch.tensor(wco2, dtype=torch.float32)
            o2 = torch.tensor(o2, dtype=torch.float32)
            y_sco2 = torch.tensor(y_sco2, dtype=torch.float32)
            y_wco2 = torch.tensor(y_wco2, dtype=torch.float32)
            y_o2 = torch.tensor(y_o2, dtype=torch.float32)

            # remove any rows with nans in all the data
            mask = torch.isnan(sco2).any(axis=1) | torch.isnan(wco2).any(axis=1) | torch.isnan(o2).any(axis=1) | torch.isnan(y_sco2).any(axis=1) | torch.isnan(y_wco2).any(axis=1) | torch.isnan(y_o2).any(axis=1)
            sco2 = sco2[~mask]
            wco2 = wco2[~mask]
            o2 = o2[~mask]
            y_sco2 = y_sco2[~mask]
            y_wco2 = y_wco2[~mask]
            y_o2 = y_o2[~mask]

            # if any shape length is 3 (i.e. [n_samples, 1, n_features]), remove the 1
            if len(sco2.shape) == 3:
                sco2 = sco2.squeeze(1)
                wco2 = wco2.squeeze(1)
                o2 = o2.squeeze(1)
            
            if self.normalize:
                with open(self.normalize_file, 'rb') as file:
                    norm_params = pickle.load(file)
                sco2 = (sco2 - norm_params['sco2_mean']) / (norm_params['sco2_std'] * norm_params['sco2_max'])
                wco2 = (wco2 - norm_params['wco2_mean']) / (norm_params['wco2_std'] * norm_params['wco2_max'])
                o2 = (o2 - norm_params['o2_mean']) / (norm_params['o2_std'] * norm_params['o2_max'])
                y_sco2 = (y_sco2 - norm_params['y_sco2_mean']) / (norm_params['y_sco2_std'] * norm_params['y_sco2_max'])
                y_wco2 = (y_wco2 - norm_params['y_wco2_mean']) / (norm_params['y_wco2_std'] * norm_params['y_wco2_max'])
                y_o2 = (y_o2 - norm_params['y_o2_mean']) / (norm_params['y_o2_std'] * norm_params['y_o2_max'])
            

            # concat sco2, wco2, and o2
            # X = np.concatenate([o2, wco2, sco2], axis = 1)
            # y = np.concatenate([y_o2, y_wco2, y_sco2], axis = 1)
            if self.band == 'sco2':
                X = sco2
                y = y_sco2
            elif self.band == 'wco2':
                X = wco2
                y = y_wco2
            elif self.band == 'o2':
                X = o2
                y = y_o2
            elif self.band == 'all':
                X = torch.cat([o2, wco2, sco2], axis=1)
                y = torch.cat([y_o2, y_wco2, y_sco2], axis=1)
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            if self.use_convolutions:
                X = X.unsqueeze(0)
                y = y.unsqueeze(0)
                X = X.permute(1, 0, 2)
                y = y.permute(1, 0, 2)

            assert X.shape[0] == y.shape[0], "Data lengths do not match"
            if self.train_on_sim:
                # swap the X and y
                X, y = y, X
            return X, y

        else:
            # filter by outcome_flag == 1
            idx = file_data['state_var'].index('RetrievalResults/outcome_flag')
            outcome_mask = file_data['states'][:, idx] == 1
            sco2_filt = [sco2[i] for i in range(len(sco2)) if outcome_mask[i]]
            wco2_filt = [wco2[i] for i in range(len(wco2)) if outcome_mask[i]]
            o2_filt = [o2[i] for i in range(len(o2)) if outcome_mask[i]]
            y_sco2_filt = [y_sco2[i] for i in range(len(y_sco2)) if outcome_mask[i]]
            y_wco2_filt = [y_wco2[i] for i in range(len(y_wco2)) if outcome_mask[i]]
            y_o2_filt = [y_o2[i] for i in range(len(y_o2)) if outcome_mask[i]]



                    


                
                
            # make everything a torch tensor
            sco2_filt = torch.tensor(sco2_filt, dtype=torch.float32)
            wco2_filt = torch.tensor(wco2_filt, dtype=torch.float32)
            o2_filt = torch.tensor(o2_filt, dtype=torch.float32)
            y_sco2_filt = torch.tensor(y_sco2_filt, dtype=torch.float32)
            y_wco2_filt = torch.tensor(y_wco2_filt, dtype=torch.float32)
            y_o2_filt = torch.tensor(y_o2_filt, dtype=torch.float32)

            # remove any rows with nans in all the data
            mask = torch.isnan(sco2_filt).any(axis=1) | torch.isnan(wco2_filt).any(axis=1) | torch.isnan(o2_filt).any(axis=1) | torch.isnan(y_sco2_filt).any(axis=1) | torch.isnan(y_wco2_filt).any(axis=1) | torch.isnan(y_o2_filt).any(axis=1)
            sco2_filt = sco2_filt[~mask]
            wco2_filt = wco2_filt[~mask]
            o2_filt = o2_filt[~mask]
            y_sco2_filt = y_sco2_filt[~mask]
            y_wco2_filt = y_wco2_filt[~mask]
            y_o2_filt = y_o2_filt[~mask]
            
            if self.normalize:
                # obs stats for normalization form band = (band - band_mean) / (band_std * max_band)
                sco2_mean = torch.mean(sco2_filt)
                sco2_std = torch.std(sco2_filt)
                sco2_max = torch.max(torch.abs(sco2_filt - sco2_mean)/sco2_std)

                wco2_mean = torch.mean(wco2_filt)
                wco2_std = torch.std(wco2_filt)
                wco2_max = torch.max(torch.abs(wco2_filt - wco2_mean)/wco2_std)

                o2_mean = torch.mean(o2_filt)
                o2_std = torch.std(o2_filt)
                o2_max = torch.max(torch.abs(o2_filt - o2_mean)/o2_std)

                y_sco2_mean = torch.mean(y_sco2_filt)
                y_sco2_std = torch.std(y_sco2_filt)
                y_sco2_max = torch.max(torch.abs(y_sco2_filt - y_sco2_mean)/y_sco2_std)

                y_wco2_mean = torch.mean(y_wco2_filt)
                y_wco2_std = torch.std(y_wco2_filt)
                y_wco2_max = torch.max(torch.abs(y_wco2_filt - y_wco2_mean)/y_wco2_std)

                y_o2_mean = torch.mean(y_o2_filt)
                y_o2_std = torch.std(y_o2_filt)
                y_o2_max = torch.max(torch.abs(y_o2_filt - y_o2_mean)/y_o2_std)


                with open(self.normalize_file, 'wb') as file:
                    pickle.dump({'sco2_mean': sco2_mean, 'sco2_std': sco2_std, 'sco2_max': sco2_max,
                                    'wco2_mean': wco2_mean, 'wco2_std': wco2_std, 'wco2_max': wco2_max,
                                    'o2_mean': o2_mean, 'o2_std': o2_std, 'o2_max': o2_max,
                                    'y_sco2_mean': y_sco2_mean, 'y_sco2_std': y_sco2_std, 'y_sco2_max': y_sco2_max,
                                    'y_wco2_mean': y_wco2_mean, 'y_wco2_std': y_wco2_std, 'y_wco2_max': y_wco2_max,
                                    'y_o2_mean': y_o2_mean, 'y_o2_std': y_o2_std, 'y_o2_max': y_o2_max
                                    }, file)
            


            # concat sco2, wco2, and o2
            # X = np.concatenate([o2, wco2, sco2], axis = 1)
            # y = np.concatenate([y_o2, y_wco2, y_sco2], axis = 1)
            # make X a torch tensor
            if self.band == 'sco2':
                X = sco2_filt
                y = y_sco2_filt
            elif self.band == 'wco2':
                X = wco2_filt
                y = y_wco2_filt
            elif self.band == 'o2':
                X = o2_filt
                y = y_o2_filt
            elif self.band == 'all':
                X = torch.cat([o2_filt, wco2_filt, sco2_filt], axis=1)
                y = torch.cat([y_o2_filt, y_wco2_filt, y_sco2_filt], axis=1)
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            if self.use_convolutions:
                X = X.unsqueeze(0)
                y = y.unsqueeze(0)
                X = X.permute(1, 0, 2)
                y = y.permute(1, 0, 2)
            print(X.shape)
            print(y.shape)
            assert X.shape[0] == y.shape[0], "Data lengths do not match"
            if self.train_on_sim:
                # swap the X and y
                X, y = y, X
            return X, y
    # if use_convolutions is True, reshape the data to [1, number of samples, number of features]



        

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> Tensor:
        X = self.X[idx]
        y = self.y[idx]
        
        sample = {'input': X, 'target': y}

        return sample
    
    


class RetrievalDataModule(pl.LightningDataModule):
    """DataModule for OCO Xgas retrieval."""
    def __init__(self, train_files, val_files, test_files, batch_size=32, use_convolutions = False,  normalize_file='normalize.pkl'):
        super().__init__()
        """Args:
            train_files: The list of training file paths
            val_files: The list of validation file paths
            test_files: The list of test file paths
            batch_size: The batch size for the DataLoader
            num_workers: The number of workers for the DataLoader
            use_convolutions: If True, reshape the data to [1, number of samples, number of features]
            normalize_file: The file to save the normalization parameters
        """
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.use_convolutions = use_convolutions
        self.normalize_file = normalize_file

        self.train_dataset = RetrievalDataSet(self.train_files, train=True, normalize=True, use_convolutions=self.use_convolutions, normalize_file=self.normalize_file)
        self.X_train, self.Y_train = self.train_dataset.X, self.train_dataset.y
        self.val_dataset = RetrievalDataSet(self.val_files, train=True, normalize=True, use_convolutions=self.use_convolutions,normalize_file=self.normalize_file)
        self.X_val, self.Y_val = self.val_dataset.X, self.val_dataset.y
        self.test_dataset = RetrievalDataSet(self.test_files, train=False, normalize=True, use_convolutions=self.use_convolutions,normalize_file=self.normalize_file)
        self.X_test, self.Y_test = self.test_dataset.X, self.test_dataset.y


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, )
    
# TODO : Filter soudnings by outcome_flag; 0 for training, no filtering for training
class SimDiffDataModule(pl.LightningDataModule):
    """DataModule for OCO sim - obs differences."""
    def __init__(self, band, train_files, val_files, test_files, batch_size=32, use_convolutions = False, train_on_sim = False, scaled_eof = False, normalize_file='normalize.pkl'):
        super().__init__()
        """Args:
            band: The band to use X and y. Options are 'sco2', 'wco2', 'o2'
            train_files: The list of training file paths
            val_files: The list of validation file paths
            test_files: The list of test file paths
            batch_size: The batch size for the DataLoader
            num_workers: The number of workers for the DataLoader
            train_on_sim: If True, train on the simulated data
            use_convolutions: If True, reshape the data to [1, number of samples, number of features]
            scaled_eof: If True, use scaled the EOFs
            normalize_file: The file to save the normalization parameters
        """
        self.band = band
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.use_convolutions = use_convolutions
        self.train_on_sim = train_on_sim
        self.scaled_eof = scaled_eof
        self.normalize_file = normalize_file

        self.train_dataset = SimDiffDataSet(self.band, self.train_files, train=True, normalize=True, use_convolutions=self.use_convolutions, train_on_sim = self.train_on_sim,scaled_eof= self.scaled_eof,normalize_file=self.normalize_file)
        self.X_train, self.Y_train = self.train_dataset.X, self.train_dataset.y
        self.val_dataset = SimDiffDataSet(self.band, self.val_files, train=True, normalize=True, use_convolutions=self.use_convolutions,train_on_sim = self.train_on_sim,scaled_eof= self.scaled_eof,normalize_file=self.normalize_file)
        self.X_val, self.Y_val = self.val_dataset.X, self.val_dataset.y
        self.test_dataset = SimDiffDataSet(self.band, self.test_files, train=False, normalize=True, use_convolutions=self.use_convolutions,train_on_sim = self.train_on_sim,scaled_eof= self.scaled_eof,normalize_file=self.normalize_file)
        self.X_test, self.Y_test = self.test_dataset.X, self.test_dataset.y


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, )

