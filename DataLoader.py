import os
import pickle
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch

class RetrievalDataLoader(pl.LightningDataModule):
    def __init__(self, data_dir, years, state_names, label_key, batch_size=64, test_year=2021):
        super().__init__()
        self.data_dir = data_dir
        self.years = years
        self.state_names = state_names
        self.label_key = label_key
        self.batch_size = batch_size
        self.test_year = test_year
        self.processed_data = []
        self.test_data = []

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            res_sims_list, states_list, labels_list = [], [], []
            for year in self.years:
                file_name = os.path.join(self.data_dir, f'data_{year}.pkl')
                year_data = self.load_pickle(file_name)
                for sample in year_data:
                    res_sims_list.append(sample['res_sims'])
                    states_list.append({name: sample['states'][name] for name in self.state_names})
                    labels_list.append(sample['states'][self.label_key])
            self.res_sims_tensor = torch.tensor(res_sims_list)
            self.states_tensor = torch.tensor(states_list)
            self.labels_tensor = torch.tensor(labels_list)
        
        if stage == 'test' or stage is None:
            test_res_sims_list, test_states_list, test_labels_list = [], [], []
            file_name = os.path.join(self.data_dir, f'data_{self.test_year}.pkl')
            test_year_data = self.load_pickle(file_name)
            for sample in test_year_data:
                test_res_sims_list.append(sample['res_sims'])
                test_states_list.append({name: sample['states'][name] for name in self.state_names})
                test_labels_list.append(sample['states'][self.label_key])
            self.test_res_sims_tensor = torch.tensor(test_res_sims_list)
            self.test_states_tensor = torch.tensor(test_states_list)
            self.test_labels_tensor = torch.tensor(test_labels_list)

    def train_dataloader(self):
        dataset = TensorDataset(self.res_sims_tensor, self.states_tensor, self.labels_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        test_dataset = TensorDataset(self.test_res_sims_tensor, self.test_states_tensor, self.test_labels_tensor)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def load_pickle(self, file_name):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        return data
    

class DenoisingDataLoader(pl.LightningDataModule):
    def __init__(self, data_dir, years, state_names, batch_size=64, test_year=2021):
        super().__init__()
        self.data_dir = data_dir
        self.years = years
        self.state_names = state_names
        self.batch_size = batch_size
        self.test_year = test_year
        self.processed_data = []
        self.test_data = []

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            res_sims_list, states_list, res_obs_list = [], [], []
            for year in self.years:
                file_name = os.path.join(self.data_dir, f'data_{year}.pkl')
                year_data = self.load_pickle(file_name)
                for sample in year_data:
                    if sample['states'].get('outcome_flag') != 1:
                        continue
                    res_sims_list.append(sample['res_sims'])
                    states_list.append({name: sample['states'][name] for name in self.state_names})
                    res_obs_list.append(sample['res_obs'])
            self.res_sims_tensor = torch.tensor(res_sims_list)
            self.states_tensor = torch.tensor(states_list)
            self.res_obs_tensor = torch.tensor(res_obs_list)
        
        if stage == 'test' or stage is None:
            test_res_sims_list, test_states_list, test_res_obs_list = [], [], []
            file_name = os.path.join(self.data_dir, f'data_{self.test_year}.pkl')
            test_year_data = self.load_pickle(file_name)
            for sample in test_year_data:
                if sample['states'].get('outcome_flag') != 1:
                    continue
                test_res_sims_list.append(sample['res_sims'])
                test_states_list.append({name: sample['states'][name] for name in self.state_names})
                test_res_obs_list.append(sample['res_obs'])
            self.test_res_sims_tensor = torch.tensor(test_res_sims_list)
            self.test_states_tensor = torch.tensor(test_states_list)
            self.test_res_obs_tensor = torch.tensor(test_res_obs_list)

    def train_dataloader(self):
        dataset = TensorDataset(self.res_sims_tensor, self.states_tensor, self.res_obs_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        test_dataset = TensorDataset(self.test_res_sims_tensor, self.test_states_tensor, self.test_res_obs_tensor)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def load_pickle(self, file_name):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        return data

# Usage
# selected_state_names = ['state1', 'state2']  # replace with actual state names you're interested in
# data_loader_module = DenoisingDataLoader('/path/to/your/pickles', range(2000, 2020), selected_state_names)
# trainer = pl.Trainer()
# trainer.fit(model, datamodule=data_loader_module)
# trainer.test(datamodule=data_loader_module)