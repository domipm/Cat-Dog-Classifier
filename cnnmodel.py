import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

class CNN(nn.Module):
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
    
    def forward(self, x):

        # First Convolutional Layer + ReLU Activation
        x = self.conv1(x)
        print("Convolution Layer 1")
        print(x.size())
        x = F.relu(x)

        # First Pooling Layer
        x = self.pool1(x)
        print("Pooling Layer 1")
        print(x.size())

        # Second Convolutional Layer + ReLU Activation
        x = self.conv2(x)
        print("Convolution Layer 2")
        print(x.size())
        x = F.relu(x)

        # Second Pooling Layer
        x = self.pool2(x)
        print("Pooling Layer 2")
        print(x.size())

        # Flatten Tensor to Vector
        x = x.view(-1, 32*128*128)
        print("Flatten Tensor")
        print(x.size())

        # First Fully-Connected (Dense) Layer -> THIS STEP STOPS WORKING!
        x = self.dense1(x)
        x = F.relu(x)
        print("Linear Layer 1")
        print(x.size())

        # Second Fully-Connected (Dense) Layer
        x = self.dense2(x)
        x = F.relu(x)
        print("Linear Layer 2")
        print(x.size())

        # Non-Linear Activation Function
        output = F.log_softmax(x)
        
        return output

# Print model summary for an input size of (3, 512, 512)
summary(CNN(), input_size=(3, 512, 512))