import numpy as np
from PIL import Image
import imageio.v2 as imageio
import os


def gravity(mass, g_vector):
    """
    The function calculates force of gravity acting on a body of mass = mass with a given acceleration vector g_vector.
    :param mass: mass of the body
    :param g_vector: gravity acceleration vector, np.array of shape (1, 2), where 2 stands for 2 dimensions
    :return: gravity force acting on a body, np.array of shape (1, 2)
    """

    # checking whether the variables given are correct
    if not isinstance(g_vector, np.ndarray):
        raise TypeError("g_vector must be a numpy array")
    assert g_vector.shape == (1, 2) and 'g_vector must be of shape (1, 2)'

    return mass * g_vector


def magnetic_field(body_charge, body_velocity, magnetic_field_z):
    """
    The function calculates Lorentz force inserted on a body charged with body_charge in magnetic field - only
    z-component of magnetic field magnetic_field_z matters here.
    :param body_charge: electric charge of the body
    :param body_velocity: instantaneous velocity of the body, np.array of shape (1, 2)
    :param magnetic_field_z: z-component of the magnetic field
    :return: Lorentz force acting on the body, np.array of shape (1, 2)
    """

    # checking whether the variables given are correct
    if not isinstance(body_velocity, np.ndarray):
        raise TypeError("body_velocity must be a numpy array")
    assert body_velocity.shape == (1, 2) and 'body_velocity must be of shape (1, 2)'

    return (body_charge * np.array([body_velocity[0][1] * magnetic_field_z, - body_velocity[0][0] * magnetic_field_z]).
            reshape(1, 2))


def harmonic_oscillator(body_position, spring_constant, equilibrium_point):
    """
    The function calculates 2D harmonic oscillator force acting on a body at a given body position assuming a given
    spring constant = spring_constant and equilibrium point = equilibrium_point (i.e. point in which force acting on a
    body is 0)
    :param body_position: position of the body
    :param spring_constant: spring constant used to calculate the force
    :param equilibrium_point: point in which force acting on a body is 0, np.array of shape (1, 2)
    :return: 2D harmonic oscillator force acting on the body, np.array of shape (1, 2)
    """

    return -spring_constant * (body_position - equilibrium_point)


def random_initial_pos(box_size, low_starting_position_limit, high_starting_position_limit, low_starting_velocity_limit,
                       high_starting_velocity_limit, time_step):
    """
    The function generates two first initial positions of a body (necessity for Verlet integration algorithm)
    :param box_size: size of the box the body is moving in
    :param low_starting_position_limit: low limit of initial position[1], int or float in range(0,1) and smaller than high_starting_position_limit
    :param high_starting_position_limit: high limit of initial position[1], int or float in range(0,1) and larger than high_starting_position_limit
    :param low_starting_velocity_limit: low limit for initial velocity, int or float smaller than high_starting_velocity_limit
    :param high_starting_velocity_limit: high limit for initial velocity, int or float larger than low_starting_velocity_limit
    :param time_step: Verlet integration time step
    :return: position, where position = np.array([position[0], position[1]]) and position[0], position[1] are np.arrays of shape (1,2)
    """

    # Checking whether the variables given are correct
    if not isinstance(low_starting_position_limit, (int, float)):
        raise TypeError("low_starting_position_limit must be an int or float")
    if not isinstance(high_starting_position_limit, (int, float)):
        raise TypeError("high_starting_position_limit must be an int or float")
    if not isinstance(low_starting_velocity_limit, (int, float)):
        raise TypeError("low_starting_velocity_limit must be an int or float")
    if not isinstance(high_starting_velocity_limit, (int, float)):
        raise TypeError("high_starting_velocity_limit must be an int or float")
    assert low_starting_position_limit < high_starting_position_limit and ('high_starting_position_limit must be'
                                                                           'greater than low_starting_position_limit')
    assert low_starting_velocity_limit < high_starting_velocity_limit and ('high_starting_velocity_limit must be'
                                                                           'greater than low_starting_velocity_limit')

    position = np.zeros((2, 2))
    velocity_0 = (low_starting_velocity_limit + (high_starting_velocity_limit - low_starting_velocity_limit) *
                  np.random.random(2))

    position[1] = box_size * (low_starting_position_limit + (high_starting_position_limit -
                                                             low_starting_position_limit) * np.random.random(2))
    position[0] = position[1] - velocity_0 * time_step

    return position


def generate_trajectories(force_type, time_step, max_simul_steps=30, box_size=1000, body_mass=1):
    """
    The function simulates the movement of a rocket with one of forces (force_type) acting on it.
    :param force_type: str, one of ('no_force', 'gravity', 'magnetic_field', 'harmonic_oscillator')
    :param time_step: the time step used in Verlet integration algorithm
    :param max_simul_steps: maximum number of simulation steps
    :param box_size: size of the box the rocket is contained
    :param body_mass: mass of the rocket/body
    :return: x and y np.arrays for each trajectory (number_of_trajectories in total)
    """

    # Checking whether force type was chosen correctly:
    assert (force_type == 'no_force' or force_type == 'gravity' or force_type == 'magnetic_field' or
            force_type == 'harmonic_oscillator') and (
               'variable force_type has to be one of (\'no_force\', \'gravity\', '
               '\'magnetic_field\', \'harmonic_oscillator\')')

    trajectories = np.array([[], []])

    # Generating random gravity acceleration, z component of magnetic field and equilibrium point for harmonic
    # oscillator:
    g_acc = -1 + 2 * np.random.random(2)
    g_acc_norm = (g_acc * 0.5 / np.linalg.norm(g_acc)).reshape(1, 2)
    B_z = -2 + 4 * np.random.random()
    r_0 = box_size * (0.4 + 0.2 * np.random.random(2))

    # Generating random initial position:
    position = random_initial_pos(box_size=box_size, low_starting_position_limit=0.3,
                                  high_starting_position_limit=0.6, low_starting_velocity_limit=-1,
                                  high_starting_velocity_limit=1,
                                  time_step=time_step)

    for j in range(max_simul_steps - 2):

        # Choosing the proper integration force:
        if force_type == 'no_force':
            position = np.append(position, (2 * position[j + 1] - position[j]).reshape(1, 2), axis=0)

        elif force_type == 'gravity':
            position = np.append(position, 2 * position[j + 1] - position[j] +
                                 gravity(mass=body_mass, g_vector=g_acc_norm) * time_step ** 2 / body_mass, axis=0)

        elif force_type == 'magnetic_field':
            # Calculating instantaneous velocity
            velocity = ((position[j + 1] - position[j]) / (2 * time_step)).reshape(1, 2)
            position = np.append(position, 2 * position[j + 1] - position[j] +
                                 magnetic_field(body_charge=1, body_velocity=velocity, magnetic_field_z=B_z) *
                                 time_step ** 2 / body_mass, axis=0)

        else:
            position = np.append(position, 2 * position[j + 1] - position[j] +
                                 harmonic_oscillator(body_position=position[j+1].reshape(1, 2), spring_constant=5,
                                                     equilibrium_point=r_0) * time_step ** 2 / body_mass, axis=0)

        # If the rocket goes out of the box we finish the simulation:
        if position[j + 2][0] < 0 or position[j + 2][0] > box_size:
            break
        if position[j + 2][1] < 0 or position[j + 2][1] > box_size:
            break

    position = np.array([position[:, 0], position[:, 1]])

    return position


def generate_simulation_from_trajectory(x, y, box_size, save_path, img_name, simulation_directory_name,
                                        rocket_img_path=r'images/rocket.png',
                                        background_img_path=r'images/background.png', background_img_size=1000,
                                        rocket_width=100, rocket_height=50, make_gif=False, frames_number=30):
    """
    The function creates images of a simulation of a rocket moving in the background according to the x, y arrays
    containing rocket trajectory
    :param np.ndarray x: array of x component of rocket trajectory
    :param np.ndarray y: array of y component of rocket trajectory
    :param int box_size: size of the box the rocket is moving in (rocket_simulation.generate_trajectories)
    :param str save_path: the path to the folder where a folder for the images of the simulation will be created
    :param str img_name: the name that will be given to each frame of a picture of a simulation
    :param str simulation_directory_name: the name that will be given to the directory in which simulation data will be saved
    :param str rocket_img_path: the path to the image of the rocket
    :param str background_img_path: the path to the image of the background
    :param int background_img_size: the size of the image of the background (assuming it is a square) in pixels
    :param int rocket_width: the width of the rocket image in pixels
    :param int rocket_height: the height of the rocket image in pixels
    :param bool make_gif: if True make also a gif out of simulation frames in the location where the jpgs are saved
    :param int frames_number: number of frames of the simulation
    :return: None
    """

    # Path for the directory in which simulation data will be saved:
    simul_directory = save_path + '\\' + simulation_directory_name

    # Checking whether the variables given are correct
    if not isinstance(x, np.ndarray):
        raise TypeError('x must be a numpy array')
    if not isinstance(y, np.ndarray):
        raise TypeError('y must be a numpy array')
    if not isinstance(box_size, int) or box_size < 0:
        raise TypeError('box_size must be a positive integer')
    if not isinstance(save_path, str):
        raise TypeError('save_path must be a string')
    if not isinstance(img_name, str):
        raise TypeError('img_name must be a string')
    if not isinstance(simulation_directory_name, str):
        raise TypeError('simulation_directory_name must be a string')
    if not isinstance(rocket_img_path, str):
        raise TypeError('rocket_img_path must be a string')
    if not isinstance(background_img_path, str):
        raise TypeError('background_img_path must be a string')
    if not isinstance(background_img_size, int) or background_img_size < 0:
        raise TypeError('background_img_size must be a positive integer')
    if not isinstance(rocket_width, int) or rocket_width < 0:
        raise TypeError('rocket_width must be a positive integer')
    if not isinstance(rocket_height, int) or rocket_height < 0:
        raise TypeError('rocket_height must be a positive integer')
    if not isinstance(img_name, str):
        raise TypeError('img_name must be a string')
    if not isinstance(make_gif, bool):
        raise TypeError('make_gif must be a bool')
    if not isinstance(frames_number, int):
        raise TypeError('frames_number must be an int')
    if os.path.exists(simul_directory):
        raise FileExistsError(f'The directory {simulation_directory_name} already exists.')

    # Creating directory in which simulation data will be saved. It will raise error if the directory already
    # exists. Simulation snapshots will be saved in directory named: 'simulation_snapshots'
    img_save_path = simul_directory + '\\' + 'simulation_snapshots'
    os.makedirs(simul_directory, exist_ok=False)
    os.makedirs(img_save_path, exist_ok=False)

    # loading in the background and the rocket images
    background = Image.open(background_img_path)
    rocket = Image.open(rocket_img_path)

    # saving the sliced x and y arrays to retain the original trajectory:
    np.save(simul_directory + '\\' + img_name + '_x_coords.npy', x)
    np.save(simul_directory + '\\' + img_name + '_y_coords.npy', y)

    # taking every step element of x and y vector so to have the wanted number of frames
    length = len(x)
    step = int(length / frames_number)
    x = x[::step]
    y = y[::step]

    # scaling the x and y coordinates so that it fits to the background size:
    x_scaled = x * background_img_size / box_size
    y_scaled = y * background_img_size / box_size

    # Adjust y-coordinates to match the image coordinate system
    y_scaled = background_img_size - y_scaled

    # adjusting the coordinates so that PIL.image.image.paste function will place the rocket at its center:
    x_scaled = x_scaled - rocket_width // 2
    y_scaled = y_scaled - rocket_height // 2

    # list to store image file names
    image_files = []

    # creating images with the rocket at each point of the trajectory:
    i = 0
    for x_coor, y_coor in zip(x_scaled, y_scaled):
        img = background.copy()
        img.paste(rocket, (int(x_coor), int(y_coor)), rocket)
        img_save_path = simul_directory + '\\' + 'simulation_snapshots' + '\\' + img_name + f'_{i}.png'
        img.save(img_save_path)
        i = i + 1
        if make_gif:
            image_files.append(img_save_path)

    # making a gif out of the images:
    if make_gif:
        with imageio.get_writer(simul_directory + '\\' + img_name + '.gif', mode='I', duration=0.5) as writer:
            for filename in image_files:
                image = imageio.imread(filename)
                writer.append_data(image)
