from torch.utils.data import random_split,  DataLoader
import pytorch_lightning as pl
from typing import Optional


class DataLoader(pl.LightningDataModule):
    def __init__(self, dataset, batch_size: int = 32, num_workers: int = 4, val_split: float = 0.2):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def prepare_data(self):
        """
        Need to sample from the large year files and save as a single training file
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            val_len = int(len(self.dataset) * self.val_split)
            train_len = len(self.dataset) - val_len
            self.train_dataset, self.val_dataset = random_split(self.dataset, [train_len, val_len])

        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)