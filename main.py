import matplotlib.pyplot as plt
from rocket_simulation import generate_trajectories, generate_simulation_from_trajectory
from animation import animate_movement, test_circle
from visualization import visualize_trajectory

'Testing the generation of trajectories:'
# fig, ax = plt.subplots()
# # x_no_force, y_no_force = generate_trajectories('no_force', time_step=1, max_simul_steps=30, box_size=10)
# # ax.scatter(x_no_force, y_no_force, s=0.5)
#
# # x_g, y_g = generate_trajectories('gravity', time_step=1, max_simul_steps=30, box_size=10)
# # ax.scatter(x_g, y_g, s=1)
#
# # x_B, y_B  = generate_trajectories('magnetic_field', time_step=0.01, max_simul_steps=3000, box_size=10)
# # ax.scatter(x_B[::150], y_B[::150], s=1)
#
# x_harm, y_harm = generate_trajectories('harmonic_oscillator', time_step=0.01, max_simul_steps=3000,
#                                        box_size=10)
# ax.scatter(x_harm, y_harm, s=0.5)
#
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)
# ax.set_aspect('equal')
# plt.show()

'Testing the creation of animation frames from coordinates'
# x, y = test_circle(5, 5, 3, 30)
#
# #animate_movement(x, y, 10)
# save_path = r'C:\Users\Public\Desktop\Python projects\physics_guesser\simulation_frames'
# generate_simulation_from_trajectory(x, y, 10, save_path, 'test', make_gif=True)

'Testing generation of simulation from generated_trajectory - magnetic field'
# (x_B, y_B), info_dict = generate_trajectories('magnetic_field', time_step=0.01, max_simul_steps=3000, box_size=10)
# fig, ax = plt.subplots()
# ax.scatter(x_B, y_B, s=0.5)
#
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)
# ax.set_aspect('equal')
# plt.show(block='True')
#
# animate_movement(x_B, y_B, 10, interval=10)
#
# save_animation = bool(int(input('Save animation (1/0)? ')))
# if save_animation:
#     save_path = r'C:\Users\Public\Desktop\Python projects\physics_guesser\simulation_frames'
#     generate_simulation_from_trajectory(x_B, y_B, 10, save_path, 'magnetic_field_test',
#                                         'magnetic_field_test_0', make_gif=True, info_dict=info_dict
#                                         )

'Testing generation of simulation from generated_trajectory - gravity'
(x_g, y_g), info_dict = generate_trajectories('gravity', time_step=0.01, max_simul_steps=3000, box_size=10)
fig, ax = plt.subplots()
ax.scatter(x_g, y_g, s=0.5)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
plt.show(block='True')

animate_movement(x_g, y_g, 10, interval=10)

save_animation = bool(int(input('Save animation (1/0)? ')))
if save_animation:
    save_path = r'C:\Users\Public\Desktop\Python projects\physics_guesser\simulation_frames'
    generate_simulation_from_trajectory(x_g, y_g, 10, save_path, 'gravity_test',
                                        'gravity_test_0', make_gif=True, info_dict=info_dict
                                        )

'Testing the loading of saved trajectory'
# x_file = r'simulation_frames/gravity_test_0/gravity_test_x_coords.npy'
# y_file = r'simulation_frames/gravity_test_0/gravity_test_y_coords.npy'
# visualize_trajectory(x_file, y_file)
