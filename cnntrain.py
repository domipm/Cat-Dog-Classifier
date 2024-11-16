import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import v2

from datetime import datetime

from alive_progress import alive_bar

import dataloader
import cnnmodel

# Folder paths for testing and training images
train_dir = "./catdog_data/train/"
test_dir = "./catdog_data/test/"

# Path where to save model parameters
save_path = "./model_params.pt"

# Seed for random numbers (based on current time, change to set fixed seed)
rand_seed = datetime.now().timestamp()
# Set seed for image selector
random.seed(rand_seed)
# Set seed for pytorch
torch.manual_seed(rand_seed)

# Transform parameters
image_size = (64,)*2

# Define transform to perform on images (augmentation)
image_transform = v2.Compose([
                    v2.ToImage(),                               # Convert to image object
                    v2.ToDtype(torch.float32, scale=True),      # Convert to tensor
                    v2.Resize(image_size),                      # Resize images to same size
                    v2.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))   # Normalize images
                ])

# Initialize datasets
train_dataset = dataloader.CatsDogsDataset(train_dir, image_transform)
test_dataset = dataloader.CatsDogsDataset(test_dir, image_transform)

# Batchsize to load from data
batchsize = 16

# Load datasets into pytorch
train_loader = DataLoader(train_dataset, batch_size = batchsize, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batchsize, shuffle = True)

# Initiaize CNN model
model = cnnmodel.CNN(train_dataset[0][0].shape)

# Training Hyperparameters
epochs = 10
learning_rate = 0.001

# Loss Function
criterion = nn.CrossEntropyLoss()
# Optimizer (Stochastic Gradient Descent)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Loss Function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Arrays for storing loss evolution over epochs
train_loss_epoch = []
test_loss_epoch = []

# Run over multiple epochs
for epoch in range(epochs):

    # Reset train and test losses
    train_loss = 0
    test_loss = 0

    # Set model to train mode
    model.train()

    # Bar for training progress visualization
    with alive_bar(total=len(train_loader),
                   max_cols=64,
                   title="Epoch {}".format(epoch+1),
                   bar="classic", 
                   spinner=None, 
                   monitor="Batch {count}/{total}", 
                   elapsed="[{elapsed}]",
                   elapsed_end="[{elapsed}]",
                   stats=None) as bar:
        
        # Run over all batches
        for batch, (image, label) in enumerate(train_loader):

            # Reset gradients
            optimizer.zero_grad()
            # Output from model
            output = model(image)
            # Compute loss
            loss = criterion(output, label)
            # Backpropagation
            loss.backward()
            # Perform one step of optimizer
            optimizer.step()

            # Add current batch loss to train loss
            train_loss += loss.item()

            # Update progress bar
            bar()

    # Compute average training loss for current epoch
    avg_train_loss = train_loss / len(train_loader)
    train_loss_epoch.append(avg_train_loss)  
    # Print out average training loss for each epoch 
    print('- Avg. Epoch Train Loss: {:.6f}'.format(avg_train_loss)) 

    # Set model to eval mode
    model.eval()

    # Ensure no gradient is computed
    with torch.no_grad():

        for batch, (image, label) in enumerate(test_loader):
            # Output from model
            output = model(image)
            # Compute loss
            loss = criterion(output, label)
            # Add current batch loss to test loss
            test_loss += loss.item()

    # Compute average test loss for current epoch
    avg_test_loss = test_loss / len(test_loader)
    test_loss_epoch.append(avg_test_loss) 
    # Print out average training loss for each epoch 
    print('- Avg. Epoch Test  Loss: {:.6f}'.format(avg_test_loss), end="\n\n") 

# After training the model, save the parameters
torch.save(model.state_dict(), save_path)