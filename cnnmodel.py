import torch.nn as nn

from torchsummary import summary

class CNN(nn.Module):

    # Initialization function with definitions for all layers
    def __init__(self, in_shape = None):
        # Initialize parent class
        super().__init__()
        # Define all layer to be used
        self.network = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),  
                    nn.BatchNorm2d(num_features=16),                                                 
                    nn.ReLU(),                                                                       
                    nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),  
                    nn.BatchNorm2d(num_features=64),                                                
                    nn.ReLU(), 
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1), 
                    nn.BatchNorm2d(num_features=256), 
                    nn.ReLU(),
                    nn.Conv2d(in_channels=256, out_channels=516, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(num_features=516),
                    nn.ReLU(),                                                                                                                                                   
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                                
                    nn.Flatten(start_dim=1),                                                         
                    nn.LazyLinear(out_features=128),
                    nn.BatchNorm1d(num_features=128),                                                   
                    nn.ReLU(),                                                                      
                    nn.Dropout(p=0.50),                                                              
                    nn.LazyLinear(out_features=32),
                    nn.BatchNorm1d(num_features=32), 
                    nn.ReLU(),
                    nn.Dropout(p=0.25),
                    nn.LazyLinear(out_features=2)                                                   
        )
        # Print model summary if given input shape
        if in_shape != None:
            summary(self, in_shape)

        return
    
    # Forward pass of the networks
    def forward(self, x):
        # Return output from last dense layer (without any activation function, 'raw' logits)
        return self.network(x)