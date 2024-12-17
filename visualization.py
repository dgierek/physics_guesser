"""These are scripts with visualisation tools"""

import numpy as np
import matplotlib.pyplot as plt
import json


def total_energy(info_dict_path, x, y, time_step=0.01, body_mass=1):
    """
    This function loads the info_dict of a given simulation
    :param str info_dict_path: path to the info_dict of a given simulation
    :param np.ndarray x: array of x coordinates of the rocket
    :param np.ndarray y: array of y coordinates of the rocket
    :param float time_step: time step used to generate trajectory in rocket_simulation.generate_trajectories
    :param int body_mass: mass of a rocket
    :return np.ndarray containing total energy
    """

    with open(info_dict_path, 'r') as json_file:
        info_dict = json.load(json_file)

    force_type = info_dict['force_type']

    velocity_x = np.diff(x) / time_step
    velocity_y = np.diff(y) / time_step

    x = x[1:]
    y = y[1:]

    kinetic_energy = body_mass * (velocity_x**2 + velocity_y**2) / 2

    if force_type == 'no_force' or force_type == 'magnetic_field':
        return kinetic_energy
    elif force_type == 'gravity':
        g_x, g_y = np.fromstring(info_dict["g_constant"].strip("[]"), sep=" ")
        return kinetic_energy + body_mass * (g_x * x + g_y * y)
    else:
        k_x, k_y = np.fromstring(info_dict["spring_constant"].strip("[]"), sep=" ")
        x_0, y_0 = np.fromstring(info_dict["equilibrium_point"].strip("[]"), sep=" ")
        return kinetic_energy + 0.5 * (k_x * (x-x_0)**2 + k_y * (y-y_0)**2)


def visualize_trajectory(x_coord_path, y_coord_path, info_dict_path, box_size=10):
    # ADD A FUNCTIONALITY TO DRAW POTENTIAL, KINETIC AND TOTAL ENERGY DURING MOVEMENT
    """
    This function is loading x and y coordinates from .npy created by
    rocket_simulation.generate_simulation_from_trajectory function while saving the simulation and creating the plot of
    the trajectory
    :param str x_coord_path: path of the .npy file with x coordinates
    :param str y_coord_path: path of the .npy file with y coordinates
    :param str info_dict_path: path to the info_dict.txt for a given simulation
    :param int box_size: size of the box the rocket is moving in simulation (see rocket_simulation.generate_trajectories)
    """

    # Checking whether the variables given are correct
    if not isinstance(x_coord_path, str):
        raise TypeError('x_coord_path must be a string')
    if not isinstance(y_coord_path, str):
        raise TypeError('y_coord_path must be a string')
    if not isinstance(box_size, int) or box_size < 0:
        raise TypeError('box_size must be a positive integer')

    # loading x and y coordinates:
    x = np.load(x_coord_path)
    y = np.load(y_coord_path)

    # calculating total energy at each step of the simulation:
    energy = total_energy(info_dict_path, x, y)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.scatter(x, y, s=0.5)

    ax1.set_xlim(0, box_size)
    ax1.set_ylim(0, box_size)
    ax1.set_aspect('equal')

    ax2.scatter(range(len(energy)), energy, s=0.5)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Total energy')

    plt.tight_layout()
    plt.show()


def visualise_batch(batch):
    """
    This function is used to visualize the images loaded by datasets.ImageDataset
    :param batch: single batch containing 3 images. Assumed shape: torch.Size([3, 3, 64, 64])
    :return:
    """

    # Plot the images
    fig, axes = plt.subplots(1, 3)

    for i, ax in enumerate(axes):
        img = batch[i].permute(1, 2, 0).numpy()  # Convert tensor to numpy array and permute dimensions
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Image {i+1}')

    plt.show()
