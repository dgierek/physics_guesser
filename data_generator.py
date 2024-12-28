from rocket_simulation import generate_trajectories_analytically, generate_simulation_from_trajectory
import matplotlib.pyplot as plt
from visualization import visualize_trajectory_analytically

def generate_data(force_type, main_folder_path, starting_idx, number_of_simulations, show_trajectory=False,
                  ask_for_save=False, make_gif=False):
    """
    The function generates a given number of simulations for a given force_type and saves the trajectory snapshots in
    a given folder.
    :param str force_type: type of force acting on a body ('no_force', 'gravity', 'magnetic_field',
                           'harmonic_oscillator')
    :param str main_folder_path: a path to the folder 'simulation_frames'
    :param int starting_idx: starting index used for naming of folders with simulations
    :param number_of_simulations: number of generated simulations
    :param bool show_trajectory: if True show each generated trajectory on a scatterplot
    :param bool ask_for_save: if True ask whether each trajectory should be saved
    :param bool make_gif: if True makes gif out of trajectory snapshots for each trajectory
    :return: None
    """

    i = 0
    while i < number_of_simulations:
        (x, y), info_dict = (None, None), None
        # generation of trajectory:
        if force_type == 'harmonic_oscillator':
            (x, y), info_dict = generate_trajectories_analytically('harmonic_oscillator', time_step=0.01,
                                                                   max_simul_steps=6000, box_size=10)
        elif force_type == 'gravity' or force_type == 'magnetic_field':
            (x, y), info_dict = generate_trajectories_analytically(force_type, time_step=0.01, max_simul_steps=3000,
                                                                   box_size=10)
        elif force_type == 'no_force':
            (x, y), info_dict = generate_trajectories_analytically('no_force', time_step=0.01,
                                                                   max_simul_steps=5000, box_size=10)
        else:
            raise TypeError

        # showing a plot of trajectory:
        if show_trajectory:
            fig, ax = plt.subplots()
            ax.scatter(x, y, s=0.5)

            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_aspect('equal')
            plt.show(block='True')

        # asking whether to save the trajectory:
        if ask_for_save:
            save_animation = bool(int(input('Save animation (1/0)? ')))
            if save_animation:
                index = i + starting_idx
                subfolder_name = force_type + f'_{index}'
                generate_simulation_from_trajectory(x, y, 10, main_folder_path, force_type,
                                                    subfolder_name, make_gif=make_gif, info_dict=info_dict)
                x_file = fr'simulation_frames/{subfolder_name}/{force_type}_x_coords.npy'
                y_file = fr'simulation_frames/{subfolder_name}/{force_type}_y_coords.npy'
                visualize_trajectory_analytically(x_file, y_file,
                                                  fr'simulation_frames/{subfolder_name}/info_dict.txt',
                                                  show_visualisation=show_trajectory)
                print('Iteration', i, 'successful')
                i += 1

        else:
            index = i + starting_idx
            subfolder_name = force_type + f'_{index}'
            generate_simulation_from_trajectory(x, y, 10, main_folder_path, force_type,
                                                subfolder_name, make_gif=make_gif, info_dict=info_dict)
            x_file = fr'simulation_frames/{subfolder_name}/{force_type}_x_coords.npy'
            y_file = fr'simulation_frames/{subfolder_name}/{force_type}_y_coords.npy'
            visualize_trajectory_analytically(x_file, y_file,
                                              fr'simulation_frames/{subfolder_name}/info_dict.txt',
                                              show_visualisation=show_trajectory)
            print('Iteration', i, 'successful')
            i += 1


'generating data for gravity simulation:'
# generate_data('gravity', r'simulation_frames', 0, 10,
#               True, True, True)

'generating data for magnetic_field simulation:'
# generate_data('magnetic_field', r'simulation_frames', 0, 10,
#               False, False, False)

'generating data for no_field simulation:'
# generate_data('no_force', r'simulation_frames', 0, 10,
#               False, False, True)

'generating data for harmonic_oscillator simulation:'
# generate_data('harmonic_oscillator', r'simulation_frames', 0, 10,
#               False, False, True)