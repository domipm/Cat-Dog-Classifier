import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from datetime import datetime

import dataloader
import cnnmodel
        
# Folder paths for testing and training images
train_dir = "./catdog_data/train/"
test_dir = "./catdog_data/test/"
valid_dir = "./catdog_data/validation/"

# Path where to save model parameters
save_path = "./model_params.pt"

# Seed for random numbers (based on current time, change to set fixed seed)
rand_seed = datetime.now().timestamp()
# Set seed for image selector
random.seed(rand_seed)
# Set seed for pytorch
torch.manual_seed(rand_seed)

# Batchsize to load from data
batchsize = 16

# Initialize training dataset
train_dataset = dataloader.CatsDogsDataset(train_dir)
# Load dataset into pytorch
train_loader = DataLoader(train_dataset, batch_size = batchsize, shuffle = True)

# Function to show images of batch
def show_batch(dataloader, show):
    for img, _ in dataloader:
        _,ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(img,nrow=4).permute(1,2,0))
        break
    if show == True: plt.show()
# Show images in a batch
show_batch(train_loader, show = False)

# Initiaize CNN model
model = cnnmodel.CNN()
# Print-out model's summary
model.model_summary(in_size = (tuple(train_dataset[0][0].shape)))

# Training Hyperparameters
epochs = 5
learning_rate = 0.001

# Loss Function
criterion = nn.CrossEntropyLoss()
# Optimizer (Stochastic Gradient Descent)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train(dataloader, model, criterion, optimizer):

    # Set model to train mode
    model.train()

    # Run over all batches in dataloader, and enumerate for images and labels
    for batch, (X, label) in enumerate(dataloader):

        # Reset gradient to zero for every batch
        optimizer.zero_grad()
        # Compute prediction
        pred = model(X)
        # Compute loss between prediction and real label
        loss = criterion(input=pred, target=label)
        # Backpropagation to compute gradients
        loss.backward()
        # Perform one step of optimizer, adjusting weights
        optimizer.step()
        # Print progress parameters
        print('Batch {}/{}\tLoss: {:f}'.format(batch+1, len(dataloader), loss.item()))

# Run over all epochs
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}\n" + "-"*64)
    train(train_loader, model, criterion, optimizer)

# After training the model, save the parameters
torch.save(model.state_dict(), save_path)