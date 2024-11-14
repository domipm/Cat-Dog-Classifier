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
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully-Connected Dense Layers
        self.dense1 = nn.Linear(in_features=32*128*128, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=2)
        
        return
    
    # Forward pass of the networks
    def forward(self, x):

        # First Convolutional Layer + ReLU Activation
        x = self.conv1(x)
        x = F.relu(x)

        # First Pooling Layer
        x = self.pool1(x)

        # Second Convolutional Layer + ReLU Activation
        x = self.conv2(x)
        x = F.relu(x)

        # Second Pooling Layer
        x = self.pool2(x)

        # Flatten Tensor to Vector
        x = x.view(-1, 32*128*128)

        # First Fully-Connected (Dense) Layer -> THIS STEP STOPS WORKING!
        x = self.dense1(x)
        x = F.relu(x)

        # Second Fully-Connected (Dense) Layer
        x = self.dense2(x)
        x = F.relu(x)

        # Non-Linear Activation Function
        output = F.log_softmax(x, dim=1)
        
        return output
    
    def model_summary(self, in_size):

        summary(CNN(), input_size = in_size)
        print()
        return