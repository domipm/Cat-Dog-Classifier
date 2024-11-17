import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.transforms import v2

from datetime import datetime

from alive_progress import alive_bar

import dataloader
import cnnmodel

# Folder paths for testing and training images
train_dir = "./catdog_data/train/"
test_dir = "./catdog_data/test/"

# Directory for writing all the outputs, graphs, and model parameters
output_dir = "./output/"

# Seed for random numbers (based on current time, change to set fixed seed)
rand_seed = datetime.now().timestamp()
# Set seed for image selector
random.seed(rand_seed)
# Set seed for pytorch
torch.manual_seed(rand_seed)

# Transform parameters
image_size = (32,)*2
p_hflip = 0.5
p_grayscale = 0.25
p_invert = 0.15
degrees = 35

# Define transform to perform on training images (augmentation)
train_transform = v2.Compose([
                    v2.ToImage(),                               # Convert to image object
                    v2.ToDtype(torch.float32, scale=True),      # Convert to tensor
                    v2.Resize(image_size),                      # Resize images to same size
                    #v2.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),  # Normalize images
                    v2.RandomHorizontalFlip(p_hflip),
                    #v2.GaussianNoise(mean=np.random.uniform(0,0.15),
                    #                 sigma=np.random.uniform(0,0.25)),
                    #v2.ColorJitter(),
                    v2.RandomRotation(degrees),
                    #v2.RandomGrayscale(p_grayscale),
                    #v2.RandomAdjustSharpness(sharpness_factor=np.random.uniform(0,2)),
                    #v2.RandomInvert(p_invert)
                ])

# Define transform to perform on testing images (just transform to tensor)
test_transform = v2.Compose([
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Resize(image_size),
])

# Initialize datasets
train_dataset = dataloader.CatsDogsDataset(train_dir, train_transform)
test_dataset = dataloader.CatsDogsDataset(test_dir, test_transform)

# Batchsize to load from data
batchsize = 32

# Load datasets into pytorch
train_loader = DataLoader(train_dataset, batch_size = batchsize, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batchsize, shuffle = True)

# Initiaize CNN model
model = cnnmodel.CNN(train_dataset[0][0].shape)

# Training Hyperparameters
epochs = 3
learning_rate = 0.00075

# Loss Function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Arrays for storing loss evolution over epochs
train_loss_epoch = []
test_loss_epoch = []
# Arrays for storing accuracy over epochs
train_accuracy_epoch = []
test_accuracy_epoch = []

# Run over multiple epochs
for epoch in range(epochs):

    # Reset train and test losses
    train_loss = 0
    test_loss = 0
    # Reset train and test accuracy
    train_accuracy = 0
    test_accuracy = 0

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
            # Add current batch accuracy to train accuracy
            train_accuracy += (output.argmax(1) == label).float().sum()
            # Update progress bar
            bar()

    # Compute average training loss for current epoch
    avg_train_loss = train_loss / len(train_loader)
    train_loss_epoch.append(avg_train_loss)  
    # Compute averate training accuracy for current epoch
    avg_train_accuracy = 100 * train_accuracy / len(train_dataset)
    train_accuracy_epoch.append(avg_train_accuracy)
    # Print out average training loss for each epoch 
    print('- Avg. Train Loss: {:.6f}\t Avg. Train Accuracy {:.6f}'.format(avg_train_loss, avg_train_accuracy)) 

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
            # Add current batch accuracy to test accuracy
            test_accuracy += (output.argmax(1) == label).float().sum()

    # Compute average test loss for current epoch
    avg_test_loss = test_loss / len(test_loader)
    test_loss_epoch.append(avg_test_loss) 
    # Compute averate training accuracy for current epoch
    avg_test_accuracy = 100 * test_accuracy / len(test_dataset)
    test_accuracy_epoch.append(avg_test_accuracy)
    # Print out average testing loss for each epoch 
    print('- Avg. Test Loss: {:.6f}\t Avg. Test Accuracy {:.6f}'.format(avg_test_loss, avg_test_accuracy), end="\n\n") 

print("-"*64)

# After training the model, save the parameters
torch.save(model.state_dict(), output_dir + "model_params.pt")

# Write the training and testing loss and accuracy to file !
with open(output_dir + "output_logs.txt", "w") as f:
    for epoch in range(epochs):
        f.write("{}\t{}\t{}\t{}\t{}\n".format(epochs, train_loss_epoch[epoch], test_loss_epoch[epoch], train_accuracy_epoch[epoch], test_accuracy_epoch[epoch]))
f.close()

# Plot training and testing loss over time
fig, ax = plt.subplots()
ax.plot(np.arange(epochs), train_loss_epoch, label="Train loss")
ax.plot(np.arange(epochs), test_loss_epoch, label="Test loss")
ax.set_title("Train/Test Loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("Average epoch loss")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.savefig(output_dir + "loss_testtrain.png", dpi=300)
plt.show()
plt.close()

# Plot training and testing accuracy over time
fig, ax = plt.subplots()
ax.plot(np.arange(epochs), train_accuracy_epoch, label="Train accuracy")
ax.plot(np.arange(epochs), test_accuracy_epoch, label="Test accuracy")
ax.set_title("Train/Test Accuracy")
ax.set_xlabel("Epochs")
ax.set_ylabel("Average epoch accuracy [%]")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.savefig(output_dir + "accuracy_testtrain.png", dpi=300)
plt.show()
plt.close()