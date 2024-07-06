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
import lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split

class RetrievalDataSet(Dataset):
    '''OCO-2 DataSet for Xgas retrieval'''
    def __init__(self, files, train=True, normalize=True, n_test = 10000,use_convolutions=False, transform=None, normalize_file='retrieval_normalize_stats.pkl'):
        self.files = files
        self.train = train
        self.normalize = normalize
        self.n_test = n_test
        self.transform = transform
        self.normalize_file = normalize_file
        self.use_convolutions = use_convolutions
        self.X, self.y = self.load_data()

    def load_data(self):
        data = { 'sco2': [], 'wco2': [], 'o2': [], 'psurf_prior': [], 'geometry': [], 'y': [] }
        for file_path in self.files:
            try:
                with open(file_path, 'rb') as file:
                    file_data = pickle.load(file)
                    for key in ['sco2', 'wco2', 'o2']:
                        band_key = f'{key}_band_no_eof' if self.train else f'{key}_band_obs'
                        data[key].append(file_data[band_key])

                    data['psurf_prior'].append(file_data['states'][:, 0].reshape(-1, 1))
                    data['geometry'].append(file_data['states'][:, 11:15])
                    data['y'].append(file_data['states'][:, 20].reshape(-1, 1))

            except (FileNotFoundError, KeyError, pickle.UnpicklingError) as e:
                print(f"Error loading file {file_path}: {e}")
        for key in data:
            data[key] = np.vstack(data[key])
            # to torch tensor
            data[key] = torch.tensor(data[key], dtype=torch.float32)
            if len(data[key].shape) == 3:
                data[key] = data[key].squeeze(1)

        
        mask = torch.isnan(data['sco2']).any(axis=1) | torch.isnan(data['wco2']).any(axis=1) | \
               torch.isnan(data['o2']).any(axis=1) | torch.isnan(data['psurf_prior']).any(axis=1) | \
               torch.isnan(data['geometry']).any(axis=1) | torch.isnan(data['y']).any(axis=1)
        
        for key in data:
            data[key] = data[key][~mask]

        if self.normalize:
            if self.train:
                self.calculate_apply_and_save_normalization_params(data)
            else:
                self.apply_normalization(data)

        X = torch.cat([data['o2'], data['wco2'], data['sco2'], data['psurf_prior'], data['geometry']], axis=1)
        y = data['y']

        if self.use_convolutions:
            X = X.unsqueeze(0)
            y = y.unsqueeze(0)
        if not self.train:
            # set seed for reproducibility
            np.random.seed(42)
            # sample from the test set
            idx = np.random.choice(X.shape[0], self.n_test, replace=False)
            X = X[idx]
            y = y[idx]

        return X, y

    def apply_normalization(self, data):
        with open(self.normalize_file, 'rb') as file:
            norm_params = pickle.load(file)
        for key in ['sco2', 'wco2', 'o2', 'psurf_prior', 'geometry', 'y']:
            mean, std = norm_params[f'{key}_mean'], norm_params[f'{key}_std']
            data[key] = (data[key] - mean) / std

    def calculate_apply_and_save_normalization_params(self, data):
        norm_params = {}
        for key in ['sco2', 'wco2', 'o2', 'psurf_prior', 'geometry', 'y']:
            mean = torch.mean(data[key])
            std = torch.std(data[key])
            data[key] = (data[key] - mean) / std
            norm_params[f'{key}_mean'] = mean
            norm_params[f'{key}_std'] = std
        with open(self.normalize_file, 'wb') as file:
            pickle.dump(norm_params, file)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {'input': self.X[idx], 'target': self.y[idx]}

class RetrievalDataModule(pl.LightningDataModule):
    def __init__(self, train_files, test_files, batch_size=32, val_split=0.1, num_workers=4):
        super().__init__()
        self.train_files = train_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.train_dataset = None

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            train_dataset = RetrievalDataSet(self.train_files, train=True)
            val_size = int(len(train_dataset) * self.val_split)
            train_size = len(train_dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(train_dataset, [train_size, val_size])
        
        if stage in (None, 'test'):
            self.test_dataset = RetrievalDataSet(self.test_files, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def get_input_shape(self):
        if self.train_dataset is not None:
            # Access the dataset to get the shape of the inputs
            dataset = self.train_dataset.dataset  # Get the actual dataset from the Subset
            return dataset.X.shape
        else:
            raise ValueError("Training dataset is not initialized. Please run setup with stage='fit' first.")
        
    def get_target_shape(self):  
        if self.train_dataset is not None:
            # Access the dataset to get the shape of the inputs
            dataset = self.train_dataset.dataset
            return dataset.y.shape
        else:   
            raise ValueError("Training dataset is not initialized. Please run setup with stage='fit' first.")

    

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
    
    


