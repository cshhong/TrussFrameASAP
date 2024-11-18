import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium.utils.save_video import save_video
import imageio
import gymenv  # Register the custom environments with __init__.py
from tqdm import tqdm  # Import tqdm for the progress bar

import os
import logging
import h5py

from h5_utils import save_episode_hdf5, load_episode_hdf5


def video_save_trigger(n_epi):
    # if n_epi % VIDEO_INTERVAL == 0:
    #     print("Saving render!")
    #     return True 
    # else:
    #     return False
    return True


def random_rollout(env, h5f, num_eps_start=0, num_episodes=25, steps_per_episode=1000, render_dir="./"):
    """
    Run episodes with random actions and save results to an open HDF5 file.

    Parameters:
    - h5f (h5py.File): An open HDF5 file object where episode data will be saved.
    - num_eps_start (int): The starting index for episode numbering. Defaults to 0.
    - num_episodes (int): The number of episodes to run. Defaults to 25.
    - steps_per_episode (int): The maximum number of steps per episode. Defaults to 1000.
    - render_dir (str): Directory path for saving any rendering outputs. Defaults to "./".

    Output:
    - Modifies the open HDF5 file with saved episode data.

    # Returns:
    # - list that contains: each episode instance[term_eps_ind, max_disp, [(v1, v2). ...] , [frame1.type_structure.idx, ] ]
    #   - terminated episode index (int)
    #   - maximum deflection observed during the episode (float)
    #   - list of failed elements, where each failed element is represented as a tuple of vertex IDs (list of tuples)
    #   - frame_grid (np.array)
    """
    term_eps_idx = 0 # only store terminated episodes
    # Run episodes with random actions
    for episode in range(num_eps_start, num_eps_start+num_episodes):
        # add progress line
        obs = env.reset()
        
        for step in tqdm(range(steps_per_episode), desc=f"Episode {episode+1} / {num_episodes}"):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            if truncated:
                print("Truncated!")
                break
            
            # If render mode is rgb_list, save video at 
            # episode ended within inventory limit
            if terminated:
                if env.render_mode == "rgb_list":
                    save_video(
                                frames=env.get_render_list(),
                                video_folder=render_dir,
                                fps=env.metadata["render_fps"],
                                # video_length = ,
                                # name_prefix = f"Episode ",
                                episode_index = episode+1,
                                # step_starting_index=step_starting_index,
                                episode_trigger = video_save_trigger 
                    )
                
                print(f"max deflection : {env.unwrapped.max_deflection}")
                # Save data for episodes only at termination to the HDF5 file
                print(f"Saving episode {term_eps_idx} to HDF5 file.")
                save_episode_hdf5(h5f, term_eps_idx, env.unwrapped.curr_fea_graph, env.unwrapped.frames, env.unwrapped.curr_frame_grid)
                # save selective data to list 
                
                term_eps_idx += 1

                # Flush data to disk (optional - may slow down training)
                h5f.flush()

                break
        
        # Display the frames for this episode
        # display_frames_as_gif(frames, episode)

    # Close the environment
    env.close()
    

if __name__ == "__main__":

    # logging.basicConfig(filename='training.log', level=logging.INFO, format='%(message)s')
    # logger = logging.getLogger()
    # random_rollout(logger)

    # # Create the environment with the render_mode "rgb_list"
    # render_dir = os.path.join("./videos/", "temp_loadinghdf5")
    # # env = gym.make("Cantilever-v0", render_mode="rgb_list")
    # # env = gym.make("Cantilever-v0", render_mode=None)
    # env = gym.make("Cantilever-v0", render_mode="rgb_end")

    # # Number of episodes and steps per episode
    # total_episodes = 50
    # steps_per_episode = 1000

    # # Open the HDF5 file once at the beginning
    hdf5_filename = 'test_loadinghdf5.h5'
    # h5f = h5py.File(hdf5_filename, 'a')  # Use 'w' to overwrite or 'a' to append
    # try: # Ensure that the HDF5 file is properly closed, even if an exception occurs during training.
    #     #Training loop and data saving
        # random_rollout(env, 
        #                h5f, 
        #                num_eps_start = 0, # make sure this is set right!
        #                num_episodes=total_episodes, 
        #                steps_per_episode=steps_per_episode,
        #                render_dir=render_dir)
    # finally:
    #     h5f.close()

    # [TODO]
    # Testing h5 file data loading

    # load one episode of data from an HDF5 file.
    loaded_fea_graph, loaded_frames, loaded_frame_grid = load_episode_hdf5(hdf5_filename, 2)
    print(f'Loaded FEA graph : \n {loaded_fea_graph}')
    print(f'Loaded Frames : \n {loaded_frames}')
    print(f'Loaded Frame Grid : \n {loaded_frame_grid}')

    ## Render from hdf5 file 
    
    
    ## Mapping frame grids to 2D UMAP
    # load all frame grids from each episode

    # test selectively loading h5 file with low deflection episode without failure -> render individual image 

    # get all episodes that didn't fail

    # 2D UMAP of shapes (framegrid) + z axis max deformation 