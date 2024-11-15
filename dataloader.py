import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import random
import os

from PIL import Image

from datetime import datetime

from torch.utils.data import Dataset
from torchvision.transforms import v2

# Data transformation to tensorview.Image tensor
transform_to_tensor = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True)
])

# Data transformation for augmentation
transform_augment = v2.Compose([
    v2.Resize([512,512]),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandAugment(num_ops=5, magnitude=10),
])

# Custom class for loading cats and dogs dataset
class CatsDogsDataset(Dataset):

    # Initialization function for dataset
    def __init__(self, directory):
        # Set the directory (train, valid, test)
        self.directory = directory
        # List of all classes to classify (folders in directory, ignores hidden files)
        self.classes_folders = [f for f in os.listdir(directory) if not f.startswith('.')]
        # List of all classes (folders without 's')
        self.classes = []
        for obj in self.classes_folders:
            self.classes.append(obj[:-1])
        # Class-to-integer encoding
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(np.sort(self.classes))}
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
        # Transform to tensor object
        image = transform_to_tensor(image)
        # Apply augmentation
        image = transform_augment(image)
        # Convert class_type to int
        class_type = torch.tensor(self.class_to_idx[class_type], dtype=torch.long)
        # Return image and class, both tensors
        return image, class_type
    
    # Function that shows random, original image in the dataset (without augmentation transform)
    def show_random(self):
        # Pick random class (folder) and image
        rand_class = random.choice(self.classes_folders)
        rand_image = random.choice(os.listdir(os.path.join(self.directory, rand_class)))
        # Open image using PIL Image
        image = Image.open(os.path.join(self.directory, rand_class, rand_image))
        # Show image using matplotlib
        plt.title(rand_image)
        plt.axis("Off")
        plt.imshow(image)
        plt.show()
        return

    # Just to showcase how the original and randomly transformed image
    def show_random_transform(self, show = True, save = False, n_images = 3):
        # Pick random class and image
        rand_class = random.choice(self.classes_folders)
        rand_image = random.choice(os.listdir(os.path.join(self.directory, rand_class)))
        # Setup matplotlib and show/save sample figure
        _, ax = plt.subplots(nrows=1, ncols=1+n_images)
        # Read image using PIL Image
        image = Image.open(os.path.join(self.directory, rand_class, rand_image))
        # Transform image to tensor without applying augmentation
        image = transform_to_tensor(image)
        # Show original
        ax[0].imshow(image.permute(1,2,0))
        ax[0].set_title(rand_image)
        ax[0].set_title(rand_class + str(rand_image).split(".")[1] + "(Org)")
        ax[0].axis("Off")
        # Apply the transformations (a couple of times)
        image_transformed = np.empty(n_images, dtype=object)
        # Show samples
        for i in range(n_images):
            image_transformed[i] = transform_to_tensor(image)
            image_transformed[i] = transform_augment(image)
            ax[i+1].imshow(image_transformed[i].permute(1,2,0))
            ax[i+1].set_title(rand_class + str(rand_image).split(".")[1] + "(Aug)")
            ax[i+1].axis("Off")
        # Save figure
        if save == True: plt.savefig("./samples/" + rand_class.split("s")[0] + str(rand_image).split(".")[1] + ".png", bbox_inches="tight",dpi=300)
        if show == True: plt.show()
        return
    
    # Function that returns all images in dataset (their paths and names)
    def list_images(self):
        # Empty array that contains all files
        arr = []
        # Iterate over all classes
        for class_type in self.classes_folders:
            # Iterate over all files in subfolders
            for file in os.listdir(os.path.join(self.directory, class_type)):
                arr.append(file)
        return arr