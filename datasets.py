import torch
import torchvision.transforms as T
import torchvision.io as io
import glob
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def add_gaussian_noise(image, mean=0.0, std=1.0):
    """Add Gaussian noise to an image."""
    noise = torch.normal(mean=mean, std=std, size=image.shape)
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0.0, 1.0)  # Ensure pixel values remain in range [0,1]


def preprocess_images(directory='simulation_frames', force_type='gravity',
                             save_name='preloaded_gravity_dataset.pt', add_noise=False, std:float=0.01):
    """
    The function loads all the images of a given force_type and saves the preloaded dataset to
    preprocessed_training_data file.
    :param directory: directory where simulation frames are stored
    :param force_type: simulations of force_type will be taken for preloading
    :param save_name: the name under which preloaded data will be saved
    :param add_noise: if True add gaussian noise to each pixel of an image
    :param std: standard deviation of gaussian noise added to images (works only if add_noise = True)
    """
    transform = T.Resize((64, 64))
    simulation_folders = glob.glob(f"{directory}/{force_type}_*")
    all_triplets = []

    first = True
    for simulation in tqdm(simulation_folders, desc='Progress of preprocessing'):
        path = Path(simulation) / 'simulation_snapshots'
        image_files = sorted(path.glob("*.png"), key=lambda x: int(x.stem.split('_')[-1]))

        for i in range(len(image_files) - 2):
            img_paths = image_files[i:i + 3]
            images = [transform(io.read_image(str(img), mode=io.ImageReadMode.RGB).float() / 255.0) for img in
                      img_paths]
            if add_noise:
                images = [add_gaussian_noise(img, mean=0.0, std=std) for img in images]
                if first:
                    plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())
                    plt.title('Example image with added noise')
                    plt.axis("off")
                    plt.show(block=True)

                    first = False


            all_triplets.append(torch.stack(images))

    dataset_tensor = torch.stack(all_triplets)
    torch.save(dataset_tensor, 'preprocessed_training_data\\' + save_name)
    print(f"Dataset successfully saved")


def prepare_preloaded_test_data(simul_dir_path=r'simulation_frames/gravity_0'):
    """
    The function loads images from a selected simulation and saves the preloaded dataset to preprocessed_training_data
    file.
    :param simul_dir_path: directory where simulation frames are stored (it has to have simulation_snapshots directory)
    """

    transform = T.Resize((64, 64))
    all_triplets = []

    path = Path(simul_dir_path) / 'simulation_snapshots'
    image_files = sorted(path.glob("*.png"), key=lambda x: int(x.stem.split('_')[-1]))

    for i in range(len(image_files) - 2):
      img_paths = image_files[i:i + 3]
      images = [transform(io.read_image(str(img), mode=io.ImageReadMode.RGB).float() / 255.0) for img in img_paths]
      all_triplets.append(torch.stack(images))

    dataset_tensor = torch.stack(all_triplets)
    torch.save(dataset_tensor, 'preprocessed_training_data/preloaded_test_data.pt')
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


if __name__ == '__main__':

    # Dataset preloading:
    #prepare_preloaded_test_data()

    # Dataset usage example:

    #dataset = LoadedDataset(preloaded_dataset_file_name='preloaded_test_data.pt')

    # dataset = LoadedDataset()
    # dataloader = DataLoader(dataset, shuffle=True)
    #
    # first_triplet = next(iter(dataloader))
    #
    # print('Length of dataset:', dataset.__len__())
    # print('Shape of first triplet:', first_triplet.shape)
    # print('First triplet:', first_triplet)

    std = 0.5
    # Prepare dataset with white noise:
    preprocess_images(save_name=f'preloaded_gravity_dataset_with_noise_std_{std}.pt', add_noise=True, std=std)
