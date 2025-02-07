import matplotlib.pyplot as plt
from rocket_simulation import generate_trajectories_analytically, generate_simulation_from_trajectory
from animation import animate_movement, test_circle
from visualization import visualize_trajectory_analytically
import numpy as np


'Testing the creation of animation frames from coordinates'
# x, y = test_circle(5, 5, 3, 30)
#
# #animate_movement(x, y, 10)
# save_path = r'C:\Users\Public\Desktop\Python projects\physics_guesser\simulation_frames'
# generate_simulation_from_trajectory(x, y, 10, save_path, 'test', make_gif=True)

'Testing generation of simulation from generated_trajectory - magnetic field'
# (x_B, y_B), info_dict = generate_trajectories_analytically('magnetic_field', time_step=0.01,
#                                                            max_simul_steps=3000, box_size=10)
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
#     x_file = r'simulation_frames/magnetic_field_test_0/magnetic_field_test_x_coords.npy'
#     y_file = r'simulation_frames/magnetic_field_test_0/magnetic_field_test_y_coords.npy'
#     visualize_trajectory_analytically(x_file, y_file,
#                                       r'simulation_frames/magnetic_field_test_0/info_dict.txt')

'Testing generation of simulation from generated_trajectory - gravity'
(x_g, y_g), info_dict = generate_trajectories_analytically('gravity', time_step=0.01, max_simul_steps=3000, box_size=10)
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
                                        'gravity_test_0', make_gif=True, info_dict=info_dict)

x_file = r'simulation_frames/gravity_test_0/gravity_test_x_coords.npy'
y_file = r'simulation_frames/gravity_test_0/gravity_test_y_coords.npy'
visualize_trajectory_analytically(x_file, y_file, r'simulation_frames/gravity_test_0/info_dict.txt')


'Testing generation of simulation from generated_trajectory - harmonic_oscillator'
# (x_h, y_h), info_dict = generate_trajectories_analytically('harmonic_oscillator', time_step=0.01,
#                                                            max_simul_steps=6000, box_size=10)
# fig, ax = plt.subplots()
# ax.scatter(x_h, y_h, s=0.5)
#
# # getting information from info_dict:
# r_x, r_y = np.fromstring(info_dict["equilibrium_point"].strip("[]"), sep=" ")
# v_0_x, v_0_y = np.fromstring(info_dict['initial_velocity'].strip("[]"), sep=" ")
# ax.scatter(r_x, r_y, c='black', label='Equilibrium point')
# ax.legend()
#
# print(v_0_x, v_0_y)
#
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)
# ax.set_aspect('equal')
# plt.show(block='True')
#
# animate_movement(x_h, y_h, 10, interval=10)
#
# save_animation = bool(int(input('Save animation (1/0)? ')))
# if save_animation:
#     save_path = r'C:\Users\Public\Desktop\Python projects\physics_guesser\simulation_frames'
#     generate_simulation_from_trajectory(x_h, y_h, 10, save_path, 'harmonic_test',
#                                         'harmonic_test_0', make_gif=True, info_dict=info_dict)
#     x_file = r'simulation_frames/harmonic_test_0/harmonic_test_x_coords.npy'
#     y_file = r'simulation_frames/harmonic_test_0/harmonic_test_y_coords.npy'
#     visualize_trajectory_analytically(x_file, y_file, r'simulation_frames/harmonic_test_0/info_dict.txt')


'Testing generation of simulation from generated trajectory - no force:'
# (x_f, y_f), info_dict = generate_trajectories_analytically('no_force', time_step=0.01, max_simul_steps=5000,
#                                                            box_size=10)
# fig, ax = plt.subplots()
# ax.scatter(x_f, y_f, s=0.5)
#
# print(x_f[-2:], y_f[-2:])
#
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)
# ax.set_aspect('equal')
# plt.show(block='True')
#
# animate_movement(x_f, y_f, 10, interval=1)
#
# save_animation = bool(int(input('Save animation (1/0)? ')))
# if save_animation:
#     save_path = r'C:\Users\Public\Desktop\Python projects\physics_guesser\simulation_frames'
#     generate_simulation_from_trajectory(x_f, y_f, 10, save_path, 'no_force_test',
#                                         'no_force_test_0', make_gif=True, info_dict=info_dict)
#     x_file = r'simulation_frames/no_force_test_0/no_force_test_x_coords.npy'
#     y_file = r'simulation_frames/no_force_test_0/no_force_test_y_coords.npy'
#     visualize_trajectory_analytically(x_file, y_file, r'simulation_frames/no_force_test_0/info_dict.txt')


'''Testing whether shapes of data and network's number of channels is correct:'''

# from torchvision import transforms
# from datasets import ImageDataset
# from torch.utils.data import DataLoader
# from autoencoder import AutoencoderWithEvolution

# transform = transforms.Compose([
#     transforms.Resize((64, 64)),  # down sampling the resolution of the images
#     transforms.ToTensor()
# ])
# dataset = ImageDataset(directory=r'simulation_frames/gravity_test_0/simulation_snapshots', transform=transform)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#
# first_batch = next(iter(dataloader))
# first_picture = first_batch.squeeze(0)[0].unsqueeze(0)
# second_picture = first_batch.squeeze(0)[1].unsqueeze(0)
# print('first_picture.shape:', first_picture.shape)  # should print torch.Size([1, 3, 64, 64])
# print('second_picture.shape:', second_picture.shape)  # should print torch.Size([1, 3, 64, 64])
#
# test_autoencoder = AutoencoderWithEvolution()
# reconstructed_image_0, z_0, _ = test_autoencoder.forward(first_picture, None)
# reconstructed_image_1, z_1, z_2_r = test_autoencoder.forward(second_picture, z_0)
# print('reconstructed_image_0.shape:', reconstructed_image_0.shape)
# print('z_0.shape:', z_0.shape)
# print('z_1.shape:', z_1.shape)
# print('z_2_r.shape:', z_2_r.shape)

'''Testing the visualisation of single picture saved in tensor'''

# from torchvision import transforms
# from datasets import ImageDataset
# from torch.utils.data import DataLoader
# from visualization import visualize_img_from_tensor
#
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),  # down sampling the resolution of the images
#     transforms.ToTensor()
# ])
# dataset = ImageDataset(directory=r'simulation_frames/gravity_test_0/simulation_snapshots', transform=transform)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#
# first_batch = next(iter(dataloader))
# first_picture = first_batch.squeeze(0)[0].unsqueeze(0)
#
# visualize_img_from_tensor(first_picture)



