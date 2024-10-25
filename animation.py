import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
point, = ax.plot([], [], 'ro')  # 'ro' stands for red color, 'o' means the point style


# Initialization function
def init():
    point.set_data([], [])
    return point,


# Animation function
def animate(i):
    x = [i / 10]  # Moves from 0 to 10
    y = x  # Line y = x
    point.set_data(x, y)
    return point,


# Call the animator
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=20, blit=True)

plt.show()