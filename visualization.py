"""These are scripts with visualisation tools"""

import numpy as np
import matplotlib.pyplot as plt
import json


def total_energy(info_dict_path, x, y, body_mass=1):
    """
    This function loads the info_dict of a given simulation and calculates the kinetic and potential energy
    :param str info_dict_path: path to the info_dict of a given simulation
    :param np.ndarray x: array of x coordinates of the rocket
    :param np.ndarray y: array of y coordinates of the rocket
    :param int body_mass: mass of a rocket
    :return np.ndarray containing kinetic energy, np.ndarray containing potential energy
    """

    with open(info_dict_path, 'r') as json_file:
        info_dict = json.load(json_file)

    force_type = info_dict['force_type']
    time_step = info_dict['time_step']

    velocity_x = (x[2:] - x[:-2]) / (2 * time_step)
    velocity_y = (y[2:] - y[:-2]) / (2 * time_step)

    velocity_x = np.pad(velocity_x, (1, 1), 'edge')
    velocity_y = np.pad(velocity_y, (1, 1), 'edge')

    kinetic_energy = body_mass * (velocity_x**2 + velocity_y**2) / 2

    if force_type == 'no_force' or force_type == 'magnetic_field':
        return kinetic_energy, 0
    elif force_type == 'gravity':
        g_x, g_y = np.fromstring(info_dict["g_constant"].strip("[]"), sep=" ")
        return kinetic_energy, body_mass * (g_x * x + g_y * y)
    else:
        k_x, k_y = np.fromstring(info_dict["spring_constant"].strip("[]"), sep=" ")
        x_0, y_0 = np.fromstring(info_dict["equilibrium_point"].strip("[]"), sep=" ")
        return kinetic_energy, 0.5 * (k_x * (x-x_0)**2 + k_y * (y-y_0)**2)


def gravitational_kinetic_energy(g_x, g_y, v_0_x, v_0_y, time, body_mass):
    """
    This function returns analytically calculated kinetic energy of a body moving in gravitational acceleration
    :param float g_x: x component of gravitational acceleration
    :param float g_y: y component of gravitational acceleration
    :param float v_0_x: x component of instantaneous body velocity at time=0
    :param float v_0_y: y component of instantaneous body velocity at time=0
    :param np.ndarray time: array of time
    :param int body_mass: mass of a body
    :return: np.ndarray
    """
    g_dot_g = g_x ** 2 + g_y ** 2
    g_dot_v_0 = g_x * v_0_x + g_y * v_0_y
    v_0_dot_v_0 = v_0_x ** 2 + v_0_y ** 2
    return (g_dot_g * time ** 2 + 2 * time * g_dot_v_0 + v_0_dot_v_0) * body_mass / 2


def harmonic_oscillator_kinetic_energy(x_0, y_0, v_0_x, v_0_y, k_x, k_y, time, body_mass):
    """
    The function returns analytically calculated kinetic energy of a harmonic oscillator
    :param float x_0: x component of body position at time=0
    :param float y_0: y component of body position at time=0
    :param float v_0_x: x component of body instantaneous velocity at time=0
    :param float v_0_y: y component of body instantaneous velocity at time=0
    :param float k_x: x component of a spring constant
    :param float k_y: y component of a spring constant
    :param np.ndarray time: array of time
    :param int body_mass: mass of a harmonic oscillator
    :return: np.ndarray
    """

    omega_x = np.sqrt(k_x / body_mass)
    omega_y = np.sqrt(k_y / body_mass)

    v_x_dot_v_x = (-x_0 * omega_x * np.sin(omega_x * time) + v_0_x * np.cos(omega_x * time))**2
    v_y_dot_v_y = (-y_0 * omega_y * np.sin(omega_y * time) + v_0_y * np.cos(omega_y * time))**2

    return body_mass * (v_x_dot_v_x + v_y_dot_v_y) / 2


def total_energy_analytically(info_dict_path, x, y, body_mass=1):
    """
    This function loads the info_dict of a given simulation and calculates the kinetic and potential energy
    :param str info_dict_path: path to the info_dict of a given simulation
    :param np.ndarray x: array of x coordinates of the rocket
    :param np.ndarray y: array of y coordinates of the rocket
    :param int body_mass: mass of a rocket
    :return np.ndarray containing kinetic energy, np.ndarray containing potential energy
    """

    with open(info_dict_path, 'r') as json_file:
        info_dict = json.load(json_file)

    force_type = info_dict['force_type']
    time_step = info_dict['time_step']
    v_0_x, v_0_y = np.fromstring(info_dict['initial_velocity'].strip("[]"), sep=" ")
    x_0, y_0 = np.fromstring(info_dict['initial_position'].strip("[]"), sep=" ")

    # Getting the array of time:
    time = np.arange(stop=len(x)*time_step, step=time_step)

    if force_type == 'no_force' or force_type == 'magnetic_field':
        return np.full(shape=len(x), fill_value=body_mass / 2 * (v_0_x ** 2 + v_0_y ** 2)), 0
    elif force_type == 'gravity':
        g_x, g_y = np.fromstring(info_dict["g_constant"].strip("[]"), sep=" ")
        return gravitational_kinetic_energy(g_x, g_y, v_0_x, v_0_y, time, body_mass), -body_mass * (g_x * x + g_y * y)
    else:
        k_x, k_y = np.fromstring(info_dict["spring_constant"].strip("[]"), sep=" ")
        r_x, r_y = np.fromstring(info_dict["equilibrium_point"].strip("[]"), sep=" ")
        return (harmonic_oscillator_kinetic_energy(x_0, y_0, v_0_x, v_0_y, k_x, k_y, time, body_mass),
                0.5 * (k_x * (x-r_x)**2 + k_y * (y-r_y)**2))


def visualize_trajectory(x_coord_path, y_coord_path, info_dict_path, box_size=10, save_energy_plot=False):
    """
    This function is loading x and y coordinates from .npy created by
    rocket_simulation.generate_simulation_from_trajectory function while saving the simulation and creating the plot of
    the trajectory and the plot of energy
    :param str x_coord_path: path of the .npy file with x coordinates
    :param str y_coord_path: path of the .npy file with y coordinates
    :param str info_dict_path: path to the info_dict.txt for a given simulation
    :param int box_size: size of the box the rocket is moving in simulation (see rocket_simulation.generate_trajectories)
    :param bool save_energy_plot: if True save the trajectory and energy plot in the directory of simulation
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
    kinetic_energy, potential_energy = total_energy(info_dict_path, x, y)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.scatter(x, y, s=0.5)

    ax1.set_xlim(0, box_size)
    ax1.set_ylim(0, box_size)
    ax1.set_aspect('equal')

    ax2.scatter(range(len(kinetic_energy)), kinetic_energy, s=0.5, c='blue', label='$E_k$')
    if potential_energy != 0:
        ax2.scatter(range(len(kinetic_energy)), potential_energy, s=0.5, c='red', label='$E_p$')
    ax2.scatter(range(len(kinetic_energy)), kinetic_energy + potential_energy, s=0.5, c='black', label='$E_t$')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('energy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(info_dict_path[:-len(r'\info_dict.txt')] + r'\trajectory_and_energy_plot.jpg')
    plt.show()


def visualize_trajectory_analytically(x_coord_path, y_coord_path, info_dict_path, box_size=10, save_energy_plot=False):
    """
    This function is loading x and y coordinates from .npy created by
    rocket_simulation.generate_simulation_from_trajectory function while saving the simulation and creating the plot of
    the trajectory and the plot of energy using analyticall formulas
    :param str x_coord_path: path of the .npy file with x coordinates
    :param str y_coord_path: path of the .npy file with y coordinates
    :param str info_dict_path: path to the info_dict.txt for a given simulation
    :param int box_size: size of the box the rocket is moving in simulation (see rocket_simulation.generate_trajectories)
    :param bool save_energy_plot: if True save the trajectory and energy plot in the directory of simulation
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
    kinetic_energy, potential_energy = total_energy_analytically(info_dict_path, x, y, body_mass=1)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.scatter(x, y, s=0.5)

    ax1.set_xlim(0, box_size)
    ax1.set_ylim(0, box_size)
    ax1.set_aspect('equal')

    ax2.scatter(range(len(kinetic_energy)), kinetic_energy, s=0.5, c='blue', label='$E_k$')
    if not isinstance(potential_energy, int):
        ax2.scatter(range(len(kinetic_energy)), potential_energy, s=0.5, c='red', label='$E_p$')
    ax2.scatter(range(len(kinetic_energy)), kinetic_energy + potential_energy, s=0.5, c='black', label='$E_t$')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Total energy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(info_dict_path[:-len(r'\info_dict.txt')] + r'\trajectory_and_energy_plot.jpg')
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
