import torch
import os

from PIL import Image

from torch.utils.data import Dataset

from torchvision.transforms import v2

# Custom class for loading cats and dogs dataset
class CatsDogsDataset(Dataset):

    # Initialization function for dataset
    def __init__(self, directory, transform):

        # Initialize directory
        self.directory = directory
        # Find how many folders are in directory
        self.folders = [f for f in os.listdir(directory) if not f.startswith('.')]
        # List of all classes
        self.classes = [obj[:-1] for obj in self.folders]
        # Class-to-integer dictionary {dog: 0, cat: 1}
        self.class_to_label = {class_name: label for label, class_name in enumerate(self.classes)}
        # List of all images in dataset
        self.images = [ file for class_type in self.folders for file in os.listdir(os.path.join(self.directory, class_type)) ]
        # Set transform to use on images
        self.transform = transform

        return
    
    # Return size of dataset (numer of all images)
    def __len__(self):

        return len(self.images)
    
    # Get single item from dataset (for indexing dataset[i] returns i-th sample)
    def __getitem__(self, index, transformed = True, return_name = None):

        # Get the image with specificed index
        image = self.images[index]
        # Get the class of the image (image naming format "class.*")
        class_type = image.split(".")[0]
        # Convert class_type to a tensor integer object
        class_type = torch.tensor(self.class_to_label[class_type], dtype=torch.long)
        # Get the path to that image (directory/class_plural/class.*)
        image_path = os.path.join(self.directory, image.split(".")[0] + "s", image)
        # Open image using PIL.Image
        image = Image.open(image_path)
        # Transform the image according to augmentation transform
        if transformed == True: image = self.transform(image)
        # If transform not given, simply convert to tensor
        else: image = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True)
        ])(image)

        # Return image and class, both tensors
        if return_name == True: return image, class_type, self.images[index][:-4]
        return image, class_type