import random
import numpy as np
import matplotlib.pyplot as plt

import torch

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import v2

from datetime import datetime

import dataloader
import cnnmodel

# Folder paths for testing and training images
data_dir = "./catdog_data/train/"

# Path where to save visualizations
save_path = "./samples/"

# Seed for random numbers (based on current time, change to set fixed seed)
rand_seed = datetime.now().timestamp()
# Set seed for image selector
random.seed(rand_seed)
# Set seed for pytorch
torch.manual_seed(rand_seed)

# Transform parameters
image_size = (64,)*2
p_hflip = 0.5
p_grayscale = 0.25
p_invert = 0.15
degrees = 35

# Define transform to perform on images (augmentation)
image_transform = v2.Compose([
                    v2.ToImage(),                               # Convert to image object
                    v2.ToDtype(torch.float32, scale=True),      # Convert to tensor
                    v2.Resize(image_size),                      # Resize images to same size
                    v2.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),  # Normalize images
                    v2.RandomHorizontalFlip(p_hflip),
                    v2.GaussianNoise(mean=np.random.uniform(0,0.15),
                                     sigma=np.random.uniform(0,0.25)),
                    v2.ColorJitter(),
                    v2.RandomRotation(degrees),
                    v2.RandomGrayscale(p_grayscale),
                    v2.RandomAdjustSharpness(sharpness_factor=np.random.uniform(0,2)),
                    v2.RandomInvert(p_invert)
                ])

# Initialize datasets
dataset = dataloader.CatsDogsDataset(data_dir, image_transform)

# Show random image from dataset
img, _, img_name = dataset.__getitem__(np.random.randint(len(dataset)), transformed=False, return_name=True)
plt.imshow(img.permute(1,2,0))
plt.axis("Off")
plt.title(img_name)
plt.savefig(save_path + "sample_image.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# Show random image (original and transformation comparison)
rand_index = np.random.randint(len(dataset))
img_org, _, img_name = dataset.__getitem__(rand_index, transformed=False, return_name=True)
# How many transformed samples
n_images = 3
_, ax = plt.subplots(nrows=1, ncols=n_images+1)
# Plot original image
ax[0].imshow(img_org.permute(1,2,0))
ax[0].axis("Off")
ax[0].set_title(img_name + str(" (Org)"))
# Plot augmented images
for n in range(n_images):
    img_aug, _ = dataset.__getitem__(rand_index)
    ax[n+1].imshow(img_aug.permute(1,2,0))
    ax[n+1].axis("Off")
    ax[n+1].set_title(img_name + str(" (Aug)"))
plt.savefig(save_path + "sample_image_transformed.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

exit()

# Batchsize to load from data
batchsize = 16

# Load datasets into pytorch
loader = DataLoader(dataset, batch_size = batchsize, shuffle = True)

# Show random sample batch from dataloader
for img, _ in loader:
    _, ax = plt.subplots()
    ax.set_title("Sample batch")
    ax.axis("Off")
    ax.imshow(make_grid(img, nrow=4).permute(1,2,0))
    break
plt.savefig(save_path + "sample_batch.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()