import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
matplotlib.use('TkAgg')


def animate_movement(x, y):
    """
    A function for displaying an animation of a point given x and y coordinate arrays. Assuming x, y are np.arrays
    """

    # Checking the data:
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Both x_values and y_values must be numpy arrays")
    assert x.shape == y.shape and 'x and y shapes don\'t match!'

    # Set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots()
    ax.set_xlim(0.9 * x.min(), 1.1 * x.max())
    ax.set_ylim(0.9 * y.min(), 1.1 * y.max())
    point, = ax.plot([], [], 'ro')  # 'ro' stands for red color, 'o' means the point style

    def init():
        """Initialization function"""

        point.set_data([], [])
        return point,

    def animate(i):
        """Animation function"""
        x_pt = x[i]
        y_pt = y[i]
        point.set_data(x_pt, y_pt)
        return point,

    # Call the animator
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=20, blit=True)

    plt.show()
