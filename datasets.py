import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
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
    """
    The class inherits from torch.utilis.data.Dataset and loads three subsequent snapshots of the simulation from a
    given directory
    """
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png')], key=natural_sort_key)
        self.transform = transform
        super(ImageDataset, self).__init__()

    def __len__(self):
        # Return the number of samples, considering groups of 3 images
        return max(0, len(self.image_files) - 2)

    def __getitem__(self, idx):
        if idx >= len(self):  # Ensures the last batches have 3 images. It's necessary as torch.utils.data.DataLoader
            # can't deal with the indexes properly by itself. I don't understand why it is the case. It looks like it's
            # really how such exceptions should be treated - accordingly to python documentation:
            # https://docs.python.org/3/reference/datamodel.html#object.__getitem__
            raise IndexError("Index out of range")

        # Load three subsequent images
        img_paths = self.image_files[idx:idx + 3]
        images = []

        for img_path in img_paths:
            img = Image.open(os.path.join(self.directory, img_path)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # Stack images into a single tensor
        images_tensor = torch.stack(images)
        return list(range(idx, idx + 3)), images_tensor


def collate_fn(batch):
    batch_indices = []
    batch_images = []
    for idx, images in batch:
        batch_indices.extend(idx)
        batch_images.append(images)
    batch_images = torch.stack(batch_images)
    return torch.tensor(batch_indices), batch_images

# 'Usage example:'
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),  # down sampling the resolution of the images
#     transforms.ToTensor()
# ])
# dataset = ImageDataset(directory=r'simulation_frames/gravity_test_0/simulation_snapshots', transform=transform)
# dataloader_batch_1 = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
# dataloader_batch_4 = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
#
# print('Length of dataset:', dataset.__len__())
# print('Length of single batch dataloader:', len(dataloader_batch_1))
# print('Length of four batch dataloader:', len(dataloader_batch_4)) # prints 8. Probably makes sense as 32/4=8
# single_batch = next(iter(dataloader_batch_1))
# four_batch = next(iter(dataloader_batch_4))
# print('Single batch shape:', single_batch[1].shape)
# print('Four batch shape:', four_batch[1].shape)
# print('Single batch indexes:', single_batch[0])
# print('Four batch indexes:', four_batch[0])
#
# for batch in four_batch[1]:
#     visualise_batch(batch)

