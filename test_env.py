import random
import gymnasium as gym
import gymenv  # Register the custom environments with __init__.py

# from stable_baselines3.common.env_util import make_vec_env

# # Assuming cantileverenv_v0.py defines a Gym environment called Cantilever-v0
# # Make sure your custom environment is registered in Gym before this step
# env_id = "Cantilever-v0"
# # Vectorized environment for parallel execution
# vec_env = make_vec_env(
#     env_id, 
#     n_envs=1, 
#     env_kwargs={
# })  

# # Create the environment
# env = vec_env(render_mode="human")

# Create the environment using gymnasium.make
env = gym.make("Cantilever-v0", render_mode="human")

# Reset the environment
obs, info = env.reset()
print(f"    curr_fea_graph :  \n {env.curr_fea_graph}")
env.unwrapped.print_framegrid()

done = False

while not done:
    print(f'    curr_fea_graph : {env.curr_fea_graph}')
    # Manually choose an action for the agent
    print("Enter the action as 'end_bool, frame_x, frame_y' or 'end' to terminate:")
    user_input = input()

    if user_input == 'end':
        done = True
        print("Ending the episode.")
        break

    try:
        action_space = env.action_space  # Get the action space of the environment

        if isinstance(action_space, gym.spaces.Box):
            low = action_space.low  # Lower bounds for each dimension
            high = action_space.high  # Upper bounds for each dimension
            # print(f"Box action space bounds: low={low}, high={high}")
        
        env_width = high[1]
        env_height = high[2]

        # Parse the action input from the user
        end_bool, frame_x, frame_y = map(int, user_input.split(','))

        # Check the bounds of the input values
        if end_bool not in [0, 1]:
            print("Invalid input for end_bool. Please enter 0 or 1.")
            continue

        if frame_x < 0 or frame_x >= env_width:
            print(f"Invalid input for frame_x. Please enter a value between 0 and {env_width - 1}.")
            continue

        if frame_y < 0 or frame_y >= env_height:
            print(f"Invalid input for frame_y. Please enter a value between 0 and {env_height - 1}.")
            continue

        action = (end_bool, frame_x, frame_y)  # Action in the form of coordinates
    except ValueError:
        print("Invalid input. Please enter 'end' or three integers separated by commas.")
        continue

    # Step through the environment with the chosen action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Print the reward and status of the episode
    print(f"Reward: {reward}, Terminated: {terminated}, Info: {info}")
    
    # Check if the environment has terminated
    if terminated:
        print("Episode has terminated.")
        done = True

# After the episode ends
env.render()  # Final render if needed
