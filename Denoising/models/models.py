import os
import tempfile
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger


class SimDiffAE(nn.Module):
    '''
        Simple AutoEncoder
    '''
    def __init__(self, n_inputs: int, n_hidden: list[int], n_outputs: int, activation: nn.Module):
        super().__init__()
        """Args:
            n_inputs: The number of input features
            n_hidden: The number of hidden units in each layer
            latent: The number of latent features
            n_outputs: The number of output features
            activation: The nonlinear activation function
        """
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, n_hidden[0]),
            activation,
            nn.Linear(n_hidden[0], n_hidden[1]),
            activation,
            nn.Linear(n_hidden[1], n_hidden[2]),
            activation,
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden[2], n_hidden[1]),
            activation,
            nn.Linear(n_hidden[1], n_hidden[0]),
            activation,
            nn.Linear(n_hidden[0], n_outputs),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class SimDiffConvAE(nn.Module):
    '''
        Simple Conv1D AutoEncoder
    '''
    def __init__(self, n_inputs: int, n_hidden: list[int], n_outputs: int, pool_size: int, activation: nn.Module):
        super().__init__()
        """Args:
            n_inputs: The number of input features
            n_hidden: The number of hidden units in each layer
            n_outputs: The number of output features
            pool_size: The size of the pooling layer
            activation: The nonlinear activation function
        """
        self.encoder = nn.Sequential(
            nn.Conv1d(n_inputs, n_hidden[0], 3),
            activation,
            # nn.MaxPool1d(pool_size),
            nn.Conv1d(n_hidden[0], n_hidden[1], 3),
            activation,
            # nn.MaxPool1d(pool_size),
            nn.Conv1d(n_hidden[1], n_hidden[2], 3),
            activation,
            nn.MaxPool1d(pool_size),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(n_hidden[2], n_hidden[1], 3),
            activation,
            # nn.Upsample(scale_factor=pool_size),
            nn.ConvTranspose1d(n_hidden[1], n_hidden[0], 3),
            activation,
            # nn.Upsample(scale_factor=pool_size),
            nn.ConvTranspose1d(n_hidden[0], n_outputs, 3),

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    


class SimDiffUNet1D(nn.Module):
    def __init__(self, n_features: int, n_filters: int = 64, n_outputs: int = 1, pool_size: int = 2, activation: nn.Module = nn.ReLU()):
        super(SimDiffUNet1D, self).__init__()
        self.n_features = n_features
        self.n_filters = n_filters
        self.n_outputs = n_outputs
        self.pool_size = pool_size
        self.activation = activation

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv1d(n_features, n_filters, kernel_size=3, padding=1),
            activation,
            nn.MaxPool1d(pool_size)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d(n_filters, n_filters * 2, kernel_size=3, padding=1),
            activation,
            nn.MaxPool1d(pool_size)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv1d(n_filters * 2, n_filters * 4, kernel_size=3, padding=1),
            activation,
            nn.MaxPool1d(pool_size)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(n_filters * 4, n_filters * 8, kernel_size=3, padding=1),
            activation
        )

        # Decoder
        self.upconv1 = nn.ConvTranspose1d(n_filters * 8, n_filters * 4, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv1d(n_filters * 8, n_filters * 4, kernel_size=3, padding=1),
            activation
        )
        self.upconv2 = nn.ConvTranspose1d(n_filters * 4, n_filters * 2, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv1d(n_filters * 4, n_filters * 2, kernel_size=3, padding=1),
            activation
        )
        self.upconv3 = nn.ConvTranspose1d(n_filters * 2, n_filters, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv1d(n_filters * 2, n_filters, kernel_size=3, padding=1),
            activation
        )

        # Output layer
        self.output_conv = nn.Conv1d(n_filters, n_outputs, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.encoder1(x)
        c2 = self.encoder2(c1)
        c3 = self.encoder3(c2)

        # Bottleneck
        b = self.bottleneck(c3)

        # Decoder
        u1 = self._pad_to_match(self.upconv1(b), c3)
        concat1 = torch.cat([u1, c3], dim=1)
        d1 = self.decoder1(concat1)

        u2 = self._pad_to_match(self.upconv2(d1), c2)
        concat2 = torch.cat([u2, c2], dim=1)
        d2 = self.decoder2(concat2)

        u3 = self._pad_to_match(self.upconv3(d2), c1)
        concat3 = torch.cat([u3, c1], dim=1)
        d3 = self.decoder3(concat3)

        # Output
        outputs = self.output_conv(d3)
        return outputs

    def _pad_to_match(self, x, ref):
        """Pads tensor x to match the shape of tensor ref"""
        diff = ref.size(2) - x.size(2)
        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[:, :, :ref.size(2)]
        return x

class SimDiffUNetTrainer(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer):
        super(SimDiffUNetTrainer, self).__init__()
        """Args:
            model: The model to train
            loss_fn: The loss function
            optimizer: The optimizer
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target']
        y_hat = self.model(x)
        y_hat = self._pad_to_match(y_hat, y)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target']
        y_hat = self.model(x)
        y_hat = self._pad_to_match(y_hat, y)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target']
        y_hat = self.model(x)
        y_hat = self._pad_to_match(y_hat, y)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def _pad_to_match(self, x, ref):
        """Pads tensor x to match the shape of tensor ref"""
        diff = ref.size(2) - x.size(2)
        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[:, :, :ref.size(2)]
        return x


    





#TODO : Get UNET working! Make sure that the datamaodule and dataset unsqueeze X and y correctly.
class SimDiffTrainer(pl.LightningModule):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, use_convolutions: bool = False,):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.use_convolutions = use_convolutions

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
