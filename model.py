import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as ilt

import torch
import random
import os

from PIL import Image

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import v2

# Find device to use (graphic acceleration if available)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device: " + device)

# Folder paths for testing and training
train_dir = "./catdog_data/train/"
test_dir = "./catdog_data/test/"
valid_dir = "./catdog_data/validation/"

# Set seed for randomizers
rand_seed = 42
random.seed(rand_seed)

# Custom class for loading cats and dogs dataset
class CatsDogsDataset(Dataset):

    # Initialization function for dataset
    def __init__(self, directory, transform = None):

        # Set the directory (train, valid, test)
        self.directory = directory
        # List of all classes to classify (folders in directory, ignores hidden files)
        self.classes = [f for f in os.listdir(directory) if not f.startswith('.')]
        # List of all the images in dataset
        self.images = self.list_images()

        return
    
    # Return size of dataset (numer of all images)
    def __len__(self):
        return len(self.images)
    
    # Get single item from dataset (for indexing dataset[i] returns i-th sample)
    def __getitem__(self, indx):
        # Find path of image indexed (dataset directory + folder class name from file name + file)
        file = self.images[indx]
        class_type = file.split(".")[0] + "s"
        path = self.directory + class_type + "/" + file    
        # Open image using PIL (transformations?)
        image = Image.open(path) 
        # Return image and its class
        return image, class_type
    
    # Function that shows random image in the dataset
    def show_random(self):
        # Pick random class and image
        rand_class = random.choice(self.classes)
        rand_image = random.choice(self.images)
        # Full directory
        image_dir = self.directory + rand_class + "/" + rand_image
        # Read image using matplotlib image
        image = ilt.imread(image_dir)
        # Show image using matplotlib
        plt.title(rand_image)
        plt.axis("off")
        plt.imshow(image)
        plt.show()

        return
    
    # Just to showcase how the defined transformations work on a sample image
    def show_random_transform(self):
        return
    
    # Function that returns all images inside given folder
    def list_images(self):

        arr = []
        dir = []
        for class_type in self.classes:
            for file in os.listdir(self.directory + class_type + "/"):
                dir.append(self.directory + class_type + "/" + file)
                arr.append(file)

        return arr

train_dataset = CatsDogsDataset(train_dir)
train_dataset.show_random()
print(train_dataset.__len__())
train_dataset[0]