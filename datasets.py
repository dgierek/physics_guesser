import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.io as io
import glob
from pathlib import Path
from tqdm import tqdm


def preprocess_images(directory='simulation_frames', force_type='gravity',
                             save_name='preloaded_gravity_dataset.pt'):
    """
    The function loads all the images of a given force_type and saves the preloaded dataset to
    preprocessed_training_data file.
    :param directory: directory where simulation frames are stored
    :param force_type: simulations of force_type will be taken for preloading
    :param save_name: the name under which preloaded data will be saved
    """
    transform = T.Resize((64, 64))
    simulation_folders = glob.glob(f"{directory}/{force_type}_*")
    all_triplets = []

    for simulation in tqdm(simulation_folders, desc='Progress of preprocessing'):
        path = Path(simulation) / 'simulation_snapshots'
        image_files = sorted(path.glob("*.png"), key=lambda x: int(x.stem.split('_')[-1]))

        for i in range(len(image_files) - 2):
            img_paths = image_files[i:i + 3]
            images = [transform(io.read_image(str(img), mode=io.ImageReadMode.RGB).float() / 255.0) for img in img_paths]
            all_triplets.append(torch.stack(images))

    dataset_tensor = torch.stack(all_triplets)
    torch.save(dataset_tensor, 'preprocessed_training_data\\' + save_name)
    print(f"Dataset successfully saved")


class LoadedDataset(torch.utils.data.Dataset):
    def __init__(self, preloaded_dataset_file_name='preloaded_gravity_dataset.pt'):
        self.data = torch.load('preprocessed_training_data\\' + preloaded_dataset_file_name)
        super(LoadedDataset, self).__init__()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if idx >= len(self):  # Prevents accessing indices beyond dataset length
            raise IndexError("Index out of range")

        return self.data[idx]


# # Dataset preloading:
# preprocess_images()

# # Dataset usage example:
#
# dataset = LoadedDataset()
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# first_batch = next(iter(dataloader))
#
# print('Length of dataset:', dataset.__len__())
# print('Shape of first batch:', first_batch.shape)
# print('First batch:', first_batch)

