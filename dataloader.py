import numpy as np
import matplotlib.pyplot as plt

import torch
import random
import os

from PIL import Image

from datetime import datetime

from torch.utils.data import Dataset
from torchvision.transforms import v2

# Find device to use (graphic acceleration if available) (not used right now?)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device: " + device)

# Folder paths for testing and training
train_dir = "./catdog_data/train/"
test_dir = "./catdog_data/test/"
valid_dir = "./catdog_data/validation/"

# Seed for random numbers (based on current time, change to set fixed seed)
rand_seed = datetime.now().timestamp()
# Set seed for image selector
random.seed(rand_seed)
# Set seed for pytorch
torch.manual_seed(rand_seed)

# Image transformation properties
image_size   =   [512,512]
flip_prob    =   0.5
deg_range    =   [-45,45]
qual_range   =   np.random.randint(15,100)
bright_range =   [0.5,1.5]
contr_range  =   [0.5,1.5]
satr_range   =   [0.5,1.5]

# Define transform to use for data augmentation
data_transform = v2.Compose([
    v2.Resize(image_size),                      # Resize all images to the same square size
    v2.RandomHorizontalFlip(p = flip_prob),     # Horizontally flip images randomly
    v2.RandomRotation(degrees = deg_range),     # Rotate and expand the image
    v2.JPEG(quality = qual_range),              # Add JPEG compression noise randomly
    v2.ColorJitter(brightness = bright_range,   # Randomly change brightness, contrast, saturation (within reason)
                    contrast = contr_range, 
                    saturation = satr_range
                    )
])

# Data transformation used (as a last step) to get the tensor image
transform_to_tensor = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True)
])

# Custom class for loading cats and dogs dataset
class CatsDogsDataset(Dataset):

    # Initialization function for dataset
    def __init__(self, directory):
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
        class_type = file.split(".")[0]
        path = self.directory + class_type + "s/" + file    
        # Open image using PIL
        image = Image.open(path)
        # Apply augmentation transforms
        image = data_transform(image)
        # Apply image-to-tensor transform
        image = transform_to_tensor(image)
        # Return image and its class
        return image, class_type
    
    # Function that shows random image in the dataset
    def show_random(self):
        # Pick random class
        rand_class = random.choice(self.classes)
        rand_image = random.choice(os.listdir(self.directory + rand_class + "/"))
        # Read image using PIL Image
        image = Image.open(self.directory + rand_class + "/" + rand_image)
        # Show image using matplotlib
        plt.title(rand_image)
        plt.axis("Off")
        plt.imshow(image)
        plt.show()
        return

    # Just to showcase how the original and randomly transformed image
    def show_random_transform(self, show = False, save = True):
        # Pick random class and image
        rand_class = random.choice(self.classes)
        rand_image = random.choice(os.listdir(self.directory + rand_class + "/"))
        # Read image using PIL Image
        image = Image.open(self.directory + rand_class + "/" + rand_image)
        # Apply the transformations (a couple of times)
        n_transforms = 3
        image_transformed = np.empty(n_transforms, dtype=object)
        # Setup matplotlib and show/save sample figure
        fig, ax = plt.subplots(nrows=1, ncols=4)
        # Show original
        ax[0].imshow(image)
        ax[0].set_title(rand_image)
        ax[0].set_title(rand_class + str(rand_image).split(".")[1] + "(Org)")
        ax[0].axis("Off")
        # Show samples
        for i in range(n_transforms):
            image_transformed[i] = data_transform(image)
            ax[i+1].imshow(image_transformed[i])
            ax[i+1].set_title(rand_class + str(rand_image).split(".")[1] + "(Aug)")
            ax[i+1].axis("Off")
        # Save figure
        if save == True: plt.savefig("./samples/" + rand_class.split("s")[0] + str(rand_image).split(".")[1] + ".png", bbox_inches="tight",dpi=300)
        if show == True: plt.show()
        return
    
    # Function that returns all images in dataset (their paths and names)
    def list_images(self):
        # Define relevant arrays
        arr, dir = [], []
        # Iterate over all classes, and over all files in each class subfolder
        for class_type in self.classes:
            for file in os.listdir(self.directory + class_type + "/"):
                # Append file path and file to arrays
                dir.append(self.directory + class_type + "/" + file)
                arr.append(file)
        return arr