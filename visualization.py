"""These are scripts with visualisation tools"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_trajectory(x_coord_path, y_coord_path, box_size=10):
    # ADD A FUNCTIONALITY TO DRAW POTENTIAL, KINETIC AND TOTAL ENERGY DURING MOVEMENT
    """
    This function is loading x and y coordinates from .npy created by
    rocket_simulation.generate_simulation_from_trajectory function while saving the simulation and creating the plot of
    the trajectory
    :param str x_coord_path: path of the .npy file with x coordinates
    :param str y_coord_path: path of the .npy file with y coordinates
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

    fig, ax = plt.subplots()

    ax.scatter(x, y, s=0.5)

    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_aspect('equal')
    plt.show()
