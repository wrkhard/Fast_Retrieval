"""
    This file contains the Pytorch Lightning Trainer classes for the retrieval and sim to real models.
    Contact: william.r.keely<at>jpl.nasa.gov
"""


import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor




class RetrievalTrainer(pl.LightningModule):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, verbose=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return self.optimizer
    



class SimDiffTrainer(pl.LightningModule):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, verbose=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        # self.log("train_loss", loss, on_epoch=False,prog_bar=False,on_step=False)
        self.log("train_loss", loss)
        # logger

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return self.optimizer
