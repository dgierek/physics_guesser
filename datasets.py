import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from visualization import visualise_batch
import re


def natural_sort_key(s):
    """
    The function is used to sort the images in directory based on the last digit in names of the images
    :param s: name string
    :return:
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png')], key=natural_sort_key)
        self.transform = transform

    def __len__(self):
        # Return the number of samples, considering groups of 3 images
        return max(0, len(self.image_files) - 2)

    def __getitem__(self, idx):
        if idx >= len(self) - 2:  # Ensures the last batches have 3 images. I don't think its correct though
            raise IndexError("Index out of range")

        # Load three subsequent images
        img_paths = self.image_files[idx:idx+3]
        images = []

        for img_path in img_paths:
            img = Image.open(os.path.join(self.directory, img_path)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # Stack images into a single tensor
        images_tensor = torch.stack(images)
        return images_tensor


'Usage example:'
# transform = transforms.Compose([
#     transforms.Resize((64, 64)), # downsampling the resolution of the images
#     transforms.ToTensor()
# ])
# dataset = ImageDataset(directory=r'simulation_frames/gravity_test_0/simulation_snapshots', transform=transform)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#
# print('Length of dataset:', dataset.__len__())
# single_batch = next(iter(dataloader))
# print('Batch shape:', single_batch.shape)  # should print torch.Size([1, 3, 3, 64, 64])
#
# for batch in dataset:
#     visualise_batch(batch)

