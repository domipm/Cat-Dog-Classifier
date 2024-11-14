import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

from torch.utils.data import DataLoader

import dataloader
import cnnmodel

from datetime import datetime

# Folder paths for testing and training images
train_dir = "./catdog_data/train/"
test_dir = "./catdog_data/test/"
valid_dir = "./catdog_data/validation/"

# Find device to use (graphic acceleration if available) (not used right now?)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("\nUsing device: " + str(device) + "\n")

# Seed for random numbers (based on current time, change to set fixed seed)
rand_seed = datetime.now().timestamp()
# Set seed for image selector
random.seed(rand_seed)
# Set seed for pytorch
torch.manual_seed(rand_seed)

# PyTorch DataLoader parameters
batchsize = 32
workers = 0

# Initialize training dataset
train_dataset = dataloader.CatsDogsDataset(train_dir)
test_dataset = dataloader.CatsDogsDataset(test_dir)
# Load dataset into pytorch
train_loader = DataLoader(train_dataset, batch_size = batchsize, num_workers = workers, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batchsize, num_workers = workers, shuffle = False)

# Initiaize CNN model
model = cnnmodel.CNN()
# Print-out model's summary
model.model_summary(in_size = (tuple(train_dataset[0][0].shape)))