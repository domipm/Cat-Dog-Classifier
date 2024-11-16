import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary
class CNN(nn.Module):

    # Initialization function with definitions for all layers
    def __init__(self, in_shape = None):

        super().__init__()

        # Define all layer to be used
        self.network = nn.Sequential(
                       nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),   # Convolution Layer
                       nn.BatchNorm2d(num_features=16),                                                 # Batch Normalization
                       nn.ReLU(),                                                                       # ReLU Activation
                       nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                                # Max Pooling Layer
                       nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),  # Convolution Layer
                       nn.BatchNorm2d(num_features=64),                                                 # Batch Normalization
                       nn.ReLU(),                                                                       # ReLU Activation
                       nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                                # Max Pooling Layer 
                       nn.Flatten(start_dim=1),                                                         # Flatten Spatial Dimensions
                       nn.LazyLinear(out_features=16),                                                  # Dense Layer
                       nn.ReLU(),                                                                       # ReLU Activation
                       nn.Dropout(p=0.25),                                                              # Dropout Layer
                       nn.LazyLinear(out_features=2)                                                    # Dense Layer
        )

        if in_shape != None:
            summary(self, in_shape)

        return
    
    # Forward pass of the networks
    def forward(self, x):

        return self.network(x) # Return output from last dense layer (without any activation function, 'raw' logits)