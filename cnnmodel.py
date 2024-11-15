import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

class CNN(nn.Module):

    # Initialization function with definitions for all layers
    def __init__(self):

        super().__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Pooling Layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully-Connected Dense Layers
        self.dense1 = nn.LazyLinear(out_features=128)
        self.dense2 = nn.LazyLinear(out_features=2)

        return
    
    # Forward pass of the networks
    def forward(self, x):

        # First Convolutional Layer + ReLU Activation
        # (3x512x512) -> (16x512x512)
        x = self.conv1(x)
        x = F.relu(x)

        # First Pooling Layer
        # (16x512x512) -> (16x256x256)
        x = self.pool(x)

        # Second Convolutional Layer + ReLU Activation
        # (16x256x256) -> (32x256x256)
        x = self.conv2(x)
        x = F.relu(x)

        # Second Pooling Layer
        # (32x256x256) -> (32x128x128)
        x = self.pool(x)

        # Flatten Tensor to Vector
        # (32x128x128) -> [32x128x128]
        x = x.view(-1, 32*128*128)

        # First Fully-Connected (Dense) Layer
        # [32x128x128] -> [128]
        x = self.dense1(x)
        x = F.relu(x)

        # Second Fully-Connected (Dense) Layer
        # [128] -> [2]
        x = self.dense2(x)

        # Non-Linear Activation Function (on spatial dimension)
        output = F.log_softmax(x, dim=1)
        
        return output
    
    # Function to print out model's summary
    def model_summary(self, in_size):

        summary(self, input_size = in_size)
        return