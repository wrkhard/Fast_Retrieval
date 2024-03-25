from lightning import LightningDataModule
from torch.utils.data import DataLoader


import os
import pickle
import torch
from torch.utils.data import Dataset

class DenoisedDataSet(Dataset):
    def __init__(self, data_paths, years):
        """
        Custom dataset for loading the specified pickle files.
        
        :param data_paths: A list of paths to the pickle files.
        :param years: A list of years to load the data for.
        """
        self.data = []
        self.labels = []

        for year in years:
            for path in data_paths[year]:
                self.load_pickle_file(path)

    def load_pickle_file(self, file_path):
        """
        Loads and appends data from a pickle file.
        
        :param file_path: Path to the pickle file to load.
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            inputs = torch.cat([
                torch.tensor(data['wco2_no_eof'], dtype=torch.float32),
                torch.tensor(data['sco2_no_eof'], dtype=torch.float32),
                torch.tensor(data['o2_no_eof'], dtype=torch.float32),
                torch.tensor(data['state'][data['state_names'] == 'RetrievalResults/psurf'], dtype=torch.float32)
            ], dim=1)
            
            labels = torch.tensor(data['state'][data['state_names'] == 'RetrievalResults/xco2'], dtype=torch.float32)

            self.data.append(inputs)
            self.labels.append(labels)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.
        """
        return self.data[idx], self.labels[idx]
    

class DenoisedDataModule(LightningDataModule):
    def __init__(self, data_paths, train_years, test_years, batch_size=32):
        super().__init__()
        self.data_paths = data_paths
        self.train_years = train_years
        self.test_years = test_years
        self.batch_size = batch_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # Load datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = DenoisedDataSet(self.data_paths, self.train_years)
            # Assuming the validation split is done externally or you have separate years for validation
            # self.val_dataset = CustomLightningDataset(self.data_paths, self.val_years)
        
        if stage == 'test' or stage is None:
            self.test_dataset = DenoisedDataSet(self.data_paths, self.test_years)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    # Uncomment if you have a validation dataset
    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
