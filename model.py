import numpy as np
import matplotlib.pyplot as plt

import torch

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import v2

# Find device to use (graphic acceleration if available)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device: " + device)

# Custom class for loading cats and dogs dataset
class CatsDogsDataset():

    def __init__(self):

        return
    
    def __len__(self):

        return
    
    def __getitem__(self):

        return
    
# Class that defines the model architecture (Pet Network)
class PetNet():

    def __init__(self):

        return
    
# Main code...