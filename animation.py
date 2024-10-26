import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
matplotlib.use('TkAgg')


def animate_movement(x, y, interval=20, save=False, save_path=None):
    """
    A function for displaying an animation of a point given x and y coordinate arrays. Assuming x, y are np.arrays.
    interval variable specifies the delay between each frame in milliseconds. The animation can be saved in form of a
    gif if save=True. Then you can specify the path by save_path variable.
    """

    # Checking the data:
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Both x_values and y_values must be numpy arrays")
    assert x.shape == y.shape and 'x and y shapes don\'t match!'

    # Set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots()
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = (x_max - x_min) * 0.1
    y_range = (y_max - y_min) * 0.1

    ax.set_xlim(x_min - x_range, x_max + x_range)
    ax.set_ylim(y_min - y_range, y_max + y_range)

    ax.set_aspect('equal')

    point, = ax.plot([], [], 'ro')  # 'ro' stands for red color, 'o' means the point style
    line, = ax.plot([], [], 'b-')  # Line to trace the points in blue color

    def init():
        """Initialization function"""
        point.set_data([], [])
        line.set_data([], [])
        return point, line

    def animate(i):
        """Animation function"""
        x_pt = x[:i+1]  # x_pt must be a sequence
        y_pt = y[:i+1]  # y_pt must be a sequence
        point.set_data([x[i]], [y[i]])
        line.set_data(x_pt, y_pt)
        return point, line

    # Call the animator
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x_values), interval=interval, blit=True)

    if save:
        if not save_path:
            ani.save('animation.gif', writer='pillow')
        elif isinstance(save_path, str):
            ani.save(save_path + r'\animation.gif', writer='pillow')
        else:
            print('Gif wasn\'t saved, because save_path is not a string')

    plt.show()


def test_circle(x_0, y_0, r, pts_density):
    """
    The function returns an np.arrays of length equal to pts_density of x and y values that represents points on the
    circle of radius r and centered around point (x_0, y_0).
    """

    theta = np.linspace(0, 2 * np.pi, pts_density)

    return x_0 + r * np.cos(theta), y_0 + r * np.sin(theta)


x_values, y_values = test_circle(5, 3, 5, 100)

animate_movement(x_values, y_values)
