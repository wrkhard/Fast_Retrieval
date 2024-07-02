"""
    Backbones for Conditional Diffusion, Evidential Regression, and Deep Ensemble Retrievals and UNet for Sim to Real. 

    Contact: william.r.keely<at>jpl.nasa.gov
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import init






class LinearMLP(nn.Module):
    """Simple Linear Neural Network"""
    def __init__(self, n_inputs: int, n_hidden: list[int], n_outputs: int):
        """Args:
            n_inputs: The number of input features
            n_hidden: The number of hidden units in each layer
            n_outputs: The number of output features
        """
        super().__init__()
        self.layers = nn.ModuleList()
        n_units = [n_inputs] + n_hidden + [n_outputs]
        for i in range(len(n_units) - 1):
            self.layers.append(nn.Linear(n_units[i], n_units[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x
    

class NonlinearMLP(nn.Module):
    """Simple Nonlinear Neural Network"""
    def __init__(self, n_inputs: int, n_hidden: list[int], n_outputs: int, activation: nn.Module):
        super().__init__()
        """Args:
            n_inputs: The number of input features
            n_hidden: The number of hidden units in each layer
            n_outputs: The number of output features
            activation: The nonlinear activation function
        """
        self.layers = nn.ModuleList()
        self.activation = activation
        n_units = [n_inputs] + n_hidden + [n_outputs]
        layers = []
        for i in range(len(n_units) - 1):
            layers.append(nn.Linear(n_units[i], n_units[i + 1]))
            if i < len(n_units) - 2:
                layers.append(self.activation)
                layers.append(nn.Dropout(0.01))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x
    

class Conv1DNN(nn.Module):
    '''
        Simple Conv1D Neural Network
    '''
    def __init__(self, n_inputs: int, n_hidden: list[int], n_outputs: int, pool_size: int,activation: nn.Module):
        super().__init__()
        """Args:
            n_inputs: The number of input features
            n_hidden: The number of hidden units in each layer
            n_outputs: The number of output features
            pool_size: The size of the pooling layer
            activation: The nonlinear activation function
        """
        self.layers = nn.ModuleList()
        self.activation = activation
        n_units = [n_inputs] + n_hidden + [n_outputs]
        layers = []
        for i in range(len(n_units) - 2):
            layers.append(nn.Conv1d(n_units[i], n_units[i + 1], 3))
            layers.append(self.activation)
            layers.append(nn.MaxPool1d(pool_size))
            layers.append(nn.Dropout(0.1))

        # linear output layer
        layers.append(nn.Flatten())
        layers.append(nn.Linear(n_units[-2], n_units[-1]))

        self.layers = nn.Sequential(*layers)
  
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x
    


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
    

    

class DenoisingUNET(nn.Module): # individual band version
    def __init__(self, num_filters=64, output_channels=1):
        super(DenoisingUNET, self).__init__()
        self.num_filters = num_filters
        self.output_channels = output_channels

        # Encoder (Downsampling)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=num_filters * 2, out_channels=num_filters * 4, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Bottleneck
        self.conv4 = nn.Conv1d(in_channels=num_filters * 4, out_channels=num_filters * 8, kernel_size=3, padding=1)

        # Decoder (Upsampling)
        self.upconv1 = nn.ConvTranspose1d(in_channels=num_filters * 8, out_channels=num_filters * 4, kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(in_channels=num_filters * 8, out_channels=num_filters * 4, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose1d(in_channels=num_filters * 4, out_channels=num_filters * 2, kernel_size=2, stride=2)
        self.conv6 = nn.Conv1d(in_channels=num_filters * 4, out_channels=num_filters * 2, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose1d(in_channels=num_filters * 2, out_channels=num_filters, kernel_size=2, stride=2)
        self.conv7 = nn.Conv1d(in_channels=num_filters * 2, out_channels=num_filters, kernel_size=3, padding=1)

        # Output layer
        self.output_conv = nn.Conv1d(in_channels=num_filters, out_channels=output_channels, kernel_size=1)

        self.relu = nn.ReLU()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        if torch.isnan(x).any():
            print('NaNs in input')
        
        c1 = self.relu(self.conv1(x))
        if torch.isnan(c1).any():
            print('NaNs in conv1 output')
        
        p1 = self.pool1(c1)
        c2 = self.relu(self.conv2(p1))
        if torch.isnan(c2).any():
            print('NaNs in conv2 output')
        
        p2 = self.pool2(c2)
        c3 = self.relu(self.conv3(p2))
        if torch.isnan(c3).any():
            print('NaNs in conv3 output')
        
        p3 = self.pool3(c3)

        # Bottleneck
        b = self.relu(self.conv4(p3))
        if torch.isnan(b).any():
            print('NaNs in conv4 output')

        # Decoder
        u1 = self.relu(self.upconv1(b))
        if torch.isnan(u1).any():
            print('NaNs in upconv1 output')
        
        concat1 = torch.cat([u1, c3], dim=1)
        c5 = self.relu(self.conv5(concat1))
        if torch.isnan(c5).any():
            print('NaNs in conv5 output')
        
        u2 = self.relu(self.upconv2(c5))
        if torch.isnan(u2).any():
            print('NaNs in upconv2 output')
        
        concat2 = torch.cat([u2, c2], dim=1)
        c6 = self.relu(self.conv6(concat2))
        if torch.isnan(c6).any():
            print('NaNs in conv6 output')
        
        u3 = self.relu(self.upconv3(c6))
        if torch.isnan(u3).any():
            print('NaNs in upconv3 output')
        
        concat3 = torch.cat([u3, c1], dim=1)
        c7 = self.relu(self.conv7(concat3))
        if torch.isnan(c7).any():
            print('NaNs in conv7 output')

        # Output
        outputs = self.output_conv(c7)
        if torch.isnan(outputs).any():
            print('NaNs in output')
        
        return outputs





