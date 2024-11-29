

def generate_trajectories(force_type, time_step, max_frames_number=30, image_pixel_size=1000,
                          number_of_trajectiories=100):

    """
    The function simulates the movement of a rocket with one of forces (force_type) acting on it.
    :param force_type: str, one of ('no_force', 'gravity', 'magnetic_field', '2D_harmonic_oscillator')
    :param time_step: the time step used in Verlet integration algorithm
    :param max_frames_number: maximum number of simulation steps
    :param image_pixel_size: size of the box the rocket is contained
    :param number_of_trajectiories: total number of different trajectories
    :return: x and y np.arrays for each trajectory (number_of_trajectories in total)
    """

