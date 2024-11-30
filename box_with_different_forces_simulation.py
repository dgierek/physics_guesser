import numpy as np


dimension = 2  # we work in 2D
m = 1  # mass of the ball
q = 1  # electric charge of the ball


def force(position, prev_position, box_size, dt):
    """
    The function calculates the force acting on a ball at a specific position. For now only gravity in implemented,
    later on more will be added.
    :param position: np. array of shape (1, dimension) position of the ball
    :return: np.array of shape (1, dimension) representing the force acting on a body
    """
    global dimension, m, q

    x_p, y_p = position

    # in the left upper corner there is force of gravity
    if y_p > box_size / 2 > x_p:
        # implementing force of gravity (constant g vector)
        g = np.array([[10, -5]])

        return m * g

    # in the right upper corner there is a spring centered at the right upper corner of the box
    elif y_p > box_size / 2 and x_p > box_size / 2:
        # implementing harmonic force (r_0 is the equilibrium point and k is elastic constant)
        r_0 = np.array([[box_size, box_size]])
        k = 1

        return -k * (position - r_0)

    # in the left lower corner there is a lorentz force
    elif y_p < box_size / 2 and x_p < box_size / 2:
        # implementing the Lorentz force (in 2D case only the magnetic field in the z direction gives forces in x and
        # directions) - B_z is magnetic field in the z direction and E is the electric field:
        B_z = 1
        E = np.array([[5, -10]])
        velocity = (position - prev_position) / (2 * dt)

        return q * (E + np.array([velocity[1] * B_z, - velocity[0] * B_z]))

    # in the right lower corner there is no force
    elif y_p < box_size / 2 < x_p:
        return np.zeros((1, 2))


def simulate_movement(num_iterations, velocity_0, box_size, dt):
    """
    Function simulates movement of a ball inside a box

    :param num_iterations: number of simulation iterations (int)
    :param velocity_0: starting velocity of the ball in form of a numpy array of shape (1,2)
    :param box_size: size of the square box the ball is moving in (float)
    :param dt: time step between each simulation iteration (float)
    :return: x and y np.arrays which are ball coordinates
    """

    global dimension, m

    # checking whether the variables given are correct
    if not isinstance(velocity_0, np.ndarray):
        raise TypeError("velovity_0 must be a numpy array")
    assert velocity_0.shape == (1, 2)
    if not isinstance(num_iterations, int):
        raise TypeError("num_iterations must be an integer")
    if not isinstance(box_size, float) and not isinstance(box_size, int):
        raise TypeError("box_size must be a float or an integer")
    if not isinstance(dt, float):
        raise TypeError("dt must be a float")

    # generating starting postions and position array:
    position = np.zeros((num_iterations, dimension))
    position[1] = 0.1 * box_size + np.random.random(dimension) * 0.8 * box_size  # starting position is random
    position[0] = position[1] - velocity_0 * dt
    position[1] = position[1]

    # simulating the movement:
    for i in range(0, num_iterations - 2):
        position[i + 2] = (2 * position[i + 1] - position[i] + force(position[i + 1], position[i], box_size, dt) *
                           dt ** 2 / m)

        # checking for collisions of the ball with the boundaries of the box
        for j in range(dimension):
            if position[i + 2][j] < 0 or position[i + 2][j] > box_size:
                # calculating velocity in case of a collision
                velocity = (position[i + 1] - position[i]) / (2 * dt)
                velocity[j] = - velocity[j]
                # adjusting the position after the collision
                position[i + 2][j] = position[i + 1][j] + velocity[j] * dt

    return position[:, 0], position[:, 1]


'''Example code:
import time as time
from animation import animate_movement


start_time = time.time()
x, y = simulate_movement(1000, 4 * np.array([[5, 1]]), box_size=10, dt=0.01)
end_time = time.time()
# print(x, y)

print(f"Execution time: {round(end_time - start_time, 3)} seconds")

animate_movement(x, y)
'''
