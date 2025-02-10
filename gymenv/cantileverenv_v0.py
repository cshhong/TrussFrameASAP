'''
Feb 4 2025
DONE apply action mask to rollout sample
DONE clean take action function with valid actions from get_action_mask 

****  Big Goals
DONE Random Rollout of environment
TODO Train with DQN? PPO? (Naive representation)
TODO [Evaluation of representation] - all solutions across training 
What should the representation be invarient to? elongated structures? What solutions are similar? Where is the jump?**

TODO How many high performing solutions does the algo explore? 
TODO How many low performing solutions does the algo explore? 
    (aka how efficient is it solving the problem)
    - find efficient way to log solutions so far 
TODO Going further is there a way to get diverse high performing solutions across topologies instead of highest performing?
    (thresholding)
TODO Going further from general group of high performing designs -> perturb points -> how does performance change? 
    Get diverse group of high performing designs through refinement 


[2. Anticipated Learning process]
Goals 
[TODO] Train with MCTC
[TODO] Train with Model-free (PPO)
[TODO] overlay with random UMAP for different timestep intervals to see exploration vs exploitation

- STEP 0 learns to take valid actions 
- STEP 1 learns to connect all supports to target loads
    Even if 0,1 are not efficient, complete episodes with have a connecting structure! -> collect all of these  
    (baked into environment) 
- STEP 2 learns to connect all supports to target loads with small deflection
    Expected towards the end to converge to a single solution

    Train for total of 1,000,000 episodes 
    => collect 100 solutions batches every 50,000 episodes
    => overlay batches on PCA map 
        - expect this to be scattered in the beginning (better exploration)
        - concentrated in high performing regions towards the end (better exploitation)

[DONE] set observation space to be continuous and action space to be discrete, add encode and decode obs, actions
[DONE] clean order of initialization with inputs
[DONE] test environment obs, action space with random rollout?

Resources

'''
import gymnasium as gym
# from gym.spaces import Box, Discrete
from gymnasium.spaces import Box, Discrete
# from gymnasium.spaces.graph import *

import sys
import os

# Add the current directory to sys.path (to call TrussFrameMechanics )
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# sys.path.append("/Users/chong/Dropbox/2024Fall/TrussframeASAP")  # ensure Python can locate the TrussFrameASAP module 

# Get the absolute path of the current file (cantileverenv_v0.py)
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f'Parent Directory of cantilever env v0 : {PARENT_DIR}')

from TrussFrameMechanics.trussframe import FrameShapeType, FrameStructureType, TrussFrameRL
from  TrussFrameMechanics.vertex import Vertex
from  TrussFrameMechanics.maximaledge import MaximalEdge
from  TrussFrameMechanics.feagraph import FEAGraph
import TrussFrameMechanics.generate_bc as generate_bc
import TrussFrameMechanics.pythonAsap as pythonAsap

from . import cantileverenv_convert_gymspaces as convert_gymspaces

import juliacall
# from pythonAsap import solve_truss_from_graph # for human playable version

import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


import numpy as np
import torch
import copy

import random

class CantileverEnv_0(gym.Env):
    '''
        use - gymnasium.make("CartPole-v1", render_mode="human")

        Initialize the environment with a specified observation mode.
        Observation Modes: 
        - 'frame_grid': Only use the frame grid.
        - 'fea_graph': Only use the FEA graph.
        - 'frame_graph': Only use the frame graph.
        
        Render modes:
            None : no render is computed (used in training) 

            'debug_all' : plots all steps (including those without valid actions)
            'debug_valid' : plots only steps with valid actions
            'debug_end' : plots only episode end
            'rgb_list' :  Returns a list of plots
                - wrapper, gymnasium.wrappers.RenderCollection is automatically applied during gymnasium.make(..., render_mode="rgb_list").
                - The frames collected are popped after render() is called or reset().

        State / Observation : 
            board (implicit as basis for other state representations)
                - (cell state for array size board_size_x, board_size_y) 
                - A grid where each cell has a length of 1 unit.
                - Structural nodes are created at the intersection points of the grid lines. 
                     y
                     ↑
                4    •----•----•----•----•----•----•
                     |    |    |    |    |    |    |
                3    •----•----•----•----•----•----•
                     |    |    |    |    |    |    |
                2    •----•----•----•----•----•----•
                     |    |    |    |    |    |    |
                1    •----•----•----•----•----•----•
                     |    |    |    |    |    |    |
                0    •----•----•----•----•----•----•
                     0    1    2    3    4    5    6   
                
            frame_grid
                - np.array dtype=int with size (self.frame_grid_size_y, self.frame_grid_size_x)
                - Modeled on cells of board
                - Grid cells are occupied with with TrussFrameRL objects (support and non-support) and proxy frames for external load
                - frames are created with a side length of 2 units (2 * cell) of the board.
                - Each TrussFrameRL object has properties defined by the coordinates of its centroid, which aligns with frame distance intersections on the board.
                - (external forces are represented as frames but in fea_graph apply external load to the bottom right vertex of the frame)
                - grid where each cell has value unoccupied= 0, free frame (light) = 2, support frame = 1, force = -1
                    Defined in FrameStructureType
                
                     y
                     ↑
                     •----•----•----•----•----•----•
                     |    |    |    |    |    |    |
                1    •-  -1-  -|-  -1-  -|-  -0-  -|
                     |    |    |    |    |    |    |
                     •----•----•----•----•----•----•
                     |    |    |    |    |    |    |
                0    •-  -2-  -|-  -0-  -|-  -0-  -|
                     |    |    |    |    |    |    | 
                     •----•----•----•----•----•----•
                          0         1         2         
            
            frame_graph 
                - Based on the frame_grid representation
                - Nodes represent frames and Edges represent adjacency, and relative distance to support/external force
                - TrussFrameRL objects are connected based on adjacency, all free TrussFrameRL objects are connected with external force object 

            fea_graph
                     y
                     ↑
                4    •----•----•----•----•----•----•
                     |    |    |   ||    |    |    |
                3    •----•----o----o----•----•----•
                     |    |    |    |    |    |    |
                2    •----o----o----o----•----•----•
                     |    |    |    |    |    |    |
                1    o----o----o----o----•----•----•
                     |    |    |    |    |    |    |
                0    ^----^----•----•----•----•----•
                     0    1    2    3    4    5    6   
                
                - FEAGraph Object with properties
                    vertices : A dictionary where keys are coordinates and values are Vertex objects.
                                Vertex objects have properties coordinates, id, edges, is_free, load
                    supports : list of vertex idx where the nodes in the frame are supports / pinned (as opposed to free)
                    edges : An adjacency list of tuples representing edges, where each tuple contains vertex indices.
                    maximal_edges : A dictionary where keys are directions and values are a list of MaximalEdge objects.
                                    MaximalEdge objects have properties direction, vertices
                    loads :  A list of tuples (node.id, [load.x, load.y, load.z]) 
                - Modeled on intersections of the board 
                - Nodes are Vertex objects which are modeled as structural nodes in ASAP
                - Edges connect nodes as modeled as structural elements in ASAP

        Action : 
            upon taking step action is masked by negative rewards(invalid action has large neg reward) to cell adjacent of existing frame 
            end episode boolean indicates that the structure is complete (there is no deterministic end condition)
            
            1) Absolute Coordinates (end_bool, freeframe_idx, frame_x, frame_y)
                End bool : 0,1 (False, True)
                Free Frame index : 2, 3  (light, medium) (follows FrameStructureType index)
                Frame x, y : 0 to self.frame_grid_size_x, self.frame_grid_size_y
            
            2) (decided not to use) Relative Coordinates (end_bool, frame_graph_id, left/right/top/bottom)
                - if there are multiple supports, should only be used with frame_graph state space

    
    '''
    metadata = {"render_modes": [None, "debug_all", "debug_valid", "rgb_list", "debug_end", "rgb_end"], 
                "render_fps": 1,
                "obs_modes" : ['frame_grid', 'fea_graph', 'frame_graph'],
                }


    def __init__(self,
                 render_mode = None,
                #  board_size_x=20,
                #  board_size_y=10,
                #  frame_size=2, 
                 video_save_interval_steps=500,
                 render_dir = 'render',
                 max_episode_length = 400,
                 obs_mode='frame_grid',
                 env_idx = 0,
                 ):
        # super().__init__()

        # self.board_size_x = board_size_x # likely divisable with self.frame_size
        # self.board_size_y = board_size_y # likely divisable with self.frame_size
        self.frame_size = 2 # actual frame length is sized in truss_analysis.jl
        self.board_size_x = self.frame_size * 10 # fixed size
        self.board_size_y = self.frame_size * 5 # fixed size
        self.max_episode_length = max_episode_length
        self.env_num_steps = 0 # activated taking valid action in main (not env.step!)
        self.episode_return = 0
        self.episode_length = 0
        self.env_idx = env_idx # index of environment in multi-env setting - used to init individual julia modules

        # Render
        # fig and ax updated with draw_fea_graph
        self.fig = None
        self.figsize = (15, 8)
        self.ax = None

        self.render_size = (15, 8)
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_list = [] # gathers plts to save to results in "rgb_list" mode
        self.render_dir = render_dir
        # Ensure the directory exists
        if not os.path.exists(self.render_dir):
            os.makedirs(self.render_dir)
        if self.render_mode == "rgb_end":
            self.render_counter = 0
        self.video_save_interval = video_save_interval_steps # in steps

        # Current State (Frames)
        self.frames=[] # stores TrussFrameRL objects in sequence of creation
        # self.support_frames = [] # list of TrussFrameRL objects
        
        # Calculate the size of the frame grid based on the frame size
        self.frame_grid_size_x = self.board_size_x // self.frame_size
        self.frame_grid_size_y = self.board_size_y // self.frame_size

        # Boundary Conditions
        self.frame_length_m = 3.0 # actual length of frame in meters used in truss analysis
        self.allowable_deflection = 0 # decided in generate_bc

        # Initialize current state
        self.curr_frame_grid = np.zeros((self.frame_grid_size_x, self.frame_grid_size_y), dtype=np.int64)

        # create dictionary frame values 
        self.curr_fea_graph = FEAGraph() #FEAGraph object
        self.curr_frame_graph = None # TODO graph representation of adjacent frames

        # if actions are taken outside of valid_pos, large negative reward is given.  
        self.valid_pos = set() # set of frame_grid coordinates (frame_x, frame_y) in which new frame can be placed 
        self.render_valid_action = False
        # if episode ends without being connected, large negative reward is given.
        self.target_loads_met = {} # Whether target load was reached : key is (x,y) coordinate on board value is True/False
        self.is_connected = False # whether the support is connected to (all) the target loads, 
        self.eps_end_valid = False # after applying action with end_bool=1, signify is episode end is valid, used in visualization

        self.disp_reward_scale = 1e2 # scale displacement reward to large positive reward

        # Used for Logging
        # self.support_frames = [] # 2D list of [x_frame, y_frame]
        # self.target_load_frames = [] # 2D list of [x_frame,y_frame, x_forcemag, y_forcemag, z_forcemag] 
        self.max_deflection = None # float max deflection of the structure after completion at FEA
        
        # Set current observation based on observation mode
        self.obs_mode = obs_mode  # Set the observation mode
        if self.obs_mode not in self.metadata["obs_modes"]:
            raise ValueError(f"Invalid observation mode: {self.obs_mode}. Valid modes are: {self.metadata['obs_modes']}")
        
        # Set in reset() when boundary conditions are set
        self.inventory_dict = None # dictionary of inventory in order of free frame types
        #     inventory = {
        #     FrameStructureType.LIGHT_FREE_FRAME : light_inv, # indicate no limits
        #     FrameStructureType.MEDIUM_FREE_FRAME : med_inv,
        #     # FrameStructureType.HEAVY_FREE_FRAME : *,
        # }
        self.obs_converter = None # ObservationBijectiveMapping object, used to encode and decode observations
        self.action_converter = None # ActionBijectiveMapping object, used to encode and decode actions
        self.observation_space = None
        self.single_observation_space = None
        self.action_space = None
        self.single_action_space = None

        # Set boundary conditions ; support, target load, inventory
        self.initBoundaryConditions() # set self.allowable_deflection, self.inventory_dict, self.frames, self.curr_frame_grid, self.target_loads_met, FrameStructureType.EXTERNAL_FORCE.node_load
        self.set_space_converters(self.inventory_dict) # set self.obs_converter, self.action_converter
        self.set_gym_spaces(self.obs_converter, self.action_converter) # set self.observation_space, self.action_space

        print("Initialized Cantilever Env!")
        self.render()


    def reset(self, seed=None, **kwargs):
        '''
        Create boundary condition within environment with 
            generate_bc.set_cantilever_env_framegrid(self.frame_grid_size_x)
        that returns support_frames, targetload_frames within the frame grid
        self.frames, self.valid_pos, self.curr_frame_grid, self.curr_frame_graph, self.curr_fea_graph is updated

        '''
        # print('Resetting Env!')

        self.render_list = []
        self.env_num_steps = 0
        self.episode_return = 0
        self.episode_length = 0
        
        # Reset the current state
        self.frames = []
        self.curr_frame_grid = np.zeros((self.frame_grid_size_x, self.frame_grid_size_y), dtype=np.int64)
        self.curr_fea_graph = FEAGraph() #FEAGraph object
        self.curr_frame_graph = None # TODO graph representation of adjacent frames
        self.valid_pos = set()

        # Reset the Vertex ID counter
        Vertex._id_counter = 1

        self.cantilever_length_f = 0 # used to limit random init in training function 
        # Set boundary conditions ; support, target load, inventory
        self.initBoundaryConditions() # set self.cantiliver_length_f , self.allowable_deflection, self.inventory_dict, self.frames, self.curr_frame_grid, self.target_loads_met, FrameStructureType.EXTERNAL_FORCE.node_load
        # self.set_space_converters(self.inventory_dict) # set self.obs_converter, self.action_converter
        # self.set_gym_spaces(self.obs_converter, self.action_converter) # set self.observation_space, self.action_space
        
        self.eps_end_valid = False

        self.render_valid_action = True # temporarily turn on to trigger render
        self.render()
        self.render_valid_action = False
        # print(f"valid pos : {self.valid_pos}")

        inventory_array = np.array(list(self.inventory_dict.values()), dtype=np.int64)
        obs = self.obs_converter.encode(self.curr_frame_grid, inventory_array) # encoded int value obs
        # print(f'Reset obs : {self.curr_frame_grid} \n {inventory_array} \n Encoded obs : {obs}')
        info = {} # no info to return

        return obs, info
    
    def initBoundaryConditions(self):
        '''
        Get boundary conditions (support, target location, inventory) from generate_bc.set_cantilever_env_framegrid
        Given that 
            - self.curr_frame_grid is initialized to zeros
            - self.frames is set to empty list
        Set
            - self.allowable_deflection
            - self.inventory_dict
            - self.frames (add support frame)
            - self.curr_frame_grid, self.curr_fea_graph, self.frame_graph (populate with support and target frames)
            - self.target_loads_met
            - FrameStructureType.EXTERNAL_FORCE.node_load (set load value)

        Set frame grid, frame graph, fea graph accordingly 
        Set support_frames, target_load_frames for logging and target_loads_met for checking if target is met
        '''
        # Get boundary conditions
        # support_frames : dictionary (x_frame, y_frame)  cell location within frame grid of support frames
        # targetload_frames : dictionary of ((x_frame,y_frame) : [x_forcemag, y_forcemag, z_forcemag] (force is applied in the negative y direction).
        # cantilever_length : length of cantilever in number of frames
        support_frames, targetload_frames, inventory_dict, cantilever_length_f = generate_bc.set_cantilever_env_framegrid(self.frame_grid_size_x)
        self.cantilever_length_f = cantilever_length_f
        self.allowable_deflection = self.frame_length_m * cantilever_length_f / 120 # length of cantilever(m) / 120
        self.inventory_dict = inventory_dict
        # set FrameStructureType.EXTERNAL_FORCE magnitude values 
        #TODO handle multiple target loads
        FrameStructureType.EXTERNAL_FORCE.node_load = list(targetload_frames.values())[0]

        # Logging
        # self.support_frames = [list(sf) for sf in support_frames] 
        # self.target_load_frames = [[coord[0], coord[1], force[0], force[1], force[2]] for coord, force in targetload_frames.items()]

        # Init supports and targets in curr_frame_grid according to bc
        for s_frame_coords in support_frames:
            s_board_coords = self.framegrid_to_board(*s_frame_coords) # convert from frame grid coords to board coords
            new_s_frame = TrussFrameRL(s_board_coords, type_structure=FrameStructureType.SUPPORT_FRAME)
            self.frames.append(new_s_frame)
            self.update_frame_grid(new_s_frame)
            self.update_frame_graph(new_s_frame)
            self.update_fea_graph(new_s_frame)

        for t_frame in targetload_frames.items():
            t_frame_coord, t_load_mag = t_frame # (x,y) on frame grid, magnitude in kN
            # convert from frame grid coords to board coords 
            t_center_board = self.framegrid_to_board(*t_frame_coord) # center of proxy frame
            t_load_board = (t_center_board[0]+self.frame_size//2 , t_center_board[1]-self.frame_size//2)# actual load board coordinate (bottom right of frame)
            # t_board_coord = (self.framegrid_to_board(t_frame_coord)[0] + self.frame_size//2, self.framegrid_to_board(t_frame_coord)[1] - self.frame_size//2) 
            new_t_frame = TrussFrameRL(t_center_board, type_structure=FrameStructureType.EXTERNAL_FORCE)
            self.update_frame_grid(new_t_frame)
            self.update_frame_graph(new_t_frame)
            self.update_fea_graph(new_t_frame, t_load_mag) #TODO need implementation
            self.target_loads_met[t_load_board] = False
    
    def set_space_converters(self, inventory_dict):
        '''
        set self.obs_converter, self.action_converter 
        Assume that self.inventory_dict is set
        '''
        assert isinstance(inventory_dict, dict), "inventory_dict must be a dictionary"
        # Set observation converter
        freeframe_inv_cap = [value for value in inventory_dict.values()] # get sequence of inventory level for freeframe types
        print(f'Freeframe inventory array: {freeframe_inv_cap}')  # Output: [1, 2, 3]
        self.obs_converter = convert_gymspaces.ObservationDownSamplingMapping(self.frame_grid_size_x, 
                                                                            self.frame_grid_size_y, 
                                                                            freeframe_inv_cap,
                                                                            kernel_size=3,
                                                                            stride=2)

        # Set action converter
        freeframe_idx_min, freeframe_idx_max = FrameStructureType.get_freeframe_idx_bounds()
        self.action_converter = convert_gymspaces.ActionBijectiveMapping(self.frame_grid_size_x, 
                                                                            self.frame_grid_size_y,
                                                                        freeframe_idx_min, 
                                                                        freeframe_idx_max)
        
    def set_gym_spaces(self, obs_converter, action_converter):
        '''
        set self.observation_space, self.action_space according to obs mode
        Assumes that observation and action converters are provided as input
        observation space is continuous (Box) and action space is discrete (Discrete)

        Input
            obs_converter : ObservationBijectiveMapping object
            action_converter : ActionBijectiveMapping object
        '''
        assert obs_converter is not None, "obs_converter must be an instance of ObservationBijectiveMapping"
        assert action_converter is not None, "action_converter must be an instance of ActionBijectiveMapping"

        # Set observation_space and action_space (following Gymnasium)
        if self.obs_mode == 'frame_grid':
            # Define Observations : frame grid 
            n_max = obs_converter.encoded_reduced_framegrid_max
            # self.observation_space = Discrete(n=n_obs)
            # self.observation_space = Box(low=0, high=n_obs, dtype=np.float64)
            # Bounds for the inventory values
            inventory_low = np.zeros(len(self.inventory_dict), dtype=np.float64)  # All inventory values >= 0
            inventory_high = np.array(list(self.inventory_dict.values()), dtype=np.float64)  # Upper bounds for inventory
            # Combine grid and inventory bounds
            low = np.concatenate([np.array([0], dtype=np.float64), inventory_low])
            high = np.concatenate([np.array([n_max], dtype=np.float64), inventory_high])
            # Create the Box observation space
            self.observation_space=Box(low=low, high=high, dtype=np.float64)
            # TODO edit obs_conversion according to this
            print(f'Observation Space : {self.observation_space}')
            self.single_observation_space = self.observation_space
            
            # Define Actions : end boolean, freeframe_type, frame_x, frame_y
            n_actions = action_converter._calculate_total_space_size()
            self.action_space = Discrete(n=n_actions)
            print(f'Action Space : {self.action_space}')
            self.single_action_space = self.action_space

            # Relative Coordinates? (end_bool, frame_idx, left/right/top/bottom) ---> Doesn't save much from absolute coordinates, still have to check valid action!

        elif self.obs_mode == 'fea_graph':
            print('TODO Need to implement set_gym_spaces for fea_graph!')
            pass
            # Gymnasium Composite Spaces - Graph or Dict?
            # Graph - node_features, edge_features, edge_links
            # Dict (not directly used in learning but can store human interpretable info)
        elif self.obs_mode == 'frame_graph':
            print('TODO Need to implement set_gym_spaces for frame_graph!')
            pass
    

    def step(self, action):
        '''
        Accepts an action (int) chosen by agent, computes the state, reward of the environment after applying the decoded action tuple and returns the 5-tuple (observation, reward, terminated, truncated, info).
                
        Input 
            action : encoded action int value
                (end_bool, freeframe_idx, frame_x, frame_y)
                
        Returns:
            observation, reward, terminated, truncated, info

        If action is invalid, produce large negative reaction and action is not applied to env
            registers transition (s=curr_state, a=action, s_prime=curr_state, r=-10, truncated=False)
            In theory with termination agent may learn how to connect faster?
            Not terminating creates more valid transitions!

        Invalid action is defined as
            - frame_x, frame_y not in valid position (tangent cells to existing design)
            - freeframe_idx inventory is 0 (other frame types are not used up)
            - end_bool = True but support and target loads are not connected
        '''
        
        self.episode_length += 1
        self.render_valid_action = False # used to trigger render
        reward = 0
        terminated = False
        truncated = False
        info = {}
        action_tuple = self.action_converter.decode(action) # int to action tuple
        end, freeframe_idx, frame_x, frame_y = action_tuple
        end_bool = True if end==1 else False
        
        # Apply action
        # print(f'    applying action : {action_tuple}')
        self.apply_action(action_tuple) # if end updates displacement and failed elements in curr_fea_graph

        # Using masked actions : all actions taken are valid actions 
        # if inventory is used up, episode is truncated
        if all(value == 0 for value in self.inventory_dict.values()):
            truncated = True # truncated because inventory is empty
        
        # terminate if action ends episode
        if end_bool==True:
            terminated = True
            self.eps_end_valid = True # used in render_frame to trigger displacement vis, in render to save final img
            # TODO adjust reward scheme 
            # reward += 10
            # reward += 2 / np.max(self.curr_fea_graph.displacement) # large reward for low deflection e.g. 0.5 / 0.01 = 50 #TODO displacement is length 0?
            if self.max_deflection < self.allowable_deflection:
                reward += 1 / self.max_deflection # large reward for low deflection e.g. 0.5 / 0.01 = 50
            reward -= 10*len(self.curr_fea_graph.failed_elements) # large penalty by number of failed elements 
            # store reward value for render 
        
        # if truncated and not terminated:
            # reward -= 10 # large penalty for not finishing within inventory 

        # Render frame
        self.render_valid_action = True

        # Store trajectory
        inventory_array = np.array(list(self.inventory_dict.values())) # convert inventory dictionary to array in order of free frame types
        obs = self.obs_converter.encode(self.curr_frame_grid, inventory_array)
        # self.print_framegrid()
        self.episode_return += reward
        self.render()
        # Add `final_info` for terminated or truncated episodes
        info = {}
        if terminated or truncated:
            # print("terminated or truncated!")
            info["final_info"] = {
                "episode": {
                    "reward": self.episode_return,
                    "length": self.episode_length,
                }
            }
        # print(f'Action : {action_tuple} \n Reward : {reward} \n Terminated : {terminated} \n Truncated : {truncated} \n Info : {info}')
        return obs, reward, terminated, truncated, info
        
    
    def check_is_connected(self, frame_x, frame_y):
        '''
        Used in step to temporarily forward check given action whetner overall structure is connected (support and target load)
        (this frame is not necessarily created in env)
        (Assumption that frames can only be built in adjacent cells allows us to check only most recent frame)
        Input
            frame center coordinates (frame_x, frame_y)
        '''
        # check if top-right, or top-left node of frame changes current self.target_loads_met values
        # given temporary changed values, if all are true, return True
        temp_target_loads_met = self.target_loads_met.copy()
        
        center = self.framegrid_to_board(frame_x, frame_y) # get center board coordinates
        top_right = (center[0] + self.frame_size//2, center[1] + self.frame_size//2)
        top_left = (center[0] - self.frame_size//2, center[1] + self.frame_size//2)

        for target in self.target_loads_met:
            if top_right == target:
                temp_target_loads_met[target] = True
            if top_left == target:
                temp_target_loads_met[target] = True

        # print(f"check_is_connected : {temp_target_loads_met}")
        
        # Check if all target loads are met
        if all(temp_target_loads_met.values()):
            return True
        else:
            return False
    
    def apply_action(self, valid_action_tuple):
        '''
        Used in step to apply action to current state
        Assumed that valid action has been checked, thus only used with valid actions
        Input 
            valid_action : (end_bool, freeframe_idx, frame_x, frame_y) coordinate 
        Updates frame_grid, fea_graph, curr_obs, frames, target_load_met and is_connected
        '''
        # create free TrussFrameRL at valid_action board coordinate
        end, freeframe_idx, frame_x, frame_y = valid_action_tuple
        freeframe_idx += self.action_converter.freeframe_idx_min # adjust freeframe_idx to indices in FrameStructureType
        frame_center = self.framegrid_to_board(frame_x, frame_y)
        # print(f"Applying Action with  freeframe_idx : {freeframe_idx} at {frame_center}")
        frame_structure_type = FrameStructureType.get_framestructuretype_from_idx(freeframe_idx)
        new_frame = TrussFrameRL(pos = frame_center, type_structure=frame_structure_type)
        
        # update current state 
        self.update_inventory_dict(new_frame)
        self.update_frame_grid(new_frame)
        self.update_fea_graph(new_frame)
        if end == 1: # update displacement info in fea graph if episode end
            self.update_displacement()
            # update self.max_deflection
            max_node_idx, self.max_deflection = self.curr_fea_graph.get_max_deflection()
            
        # TODO self.update_frame_graph(new_frame)

        # self.update_curr_obs()

        self.frames.append(new_frame)

        self.update_target_meet(new_frame) # updates self.target_loads_met and self.is_connected
    
    ## Update Current State
    def update_inventory_dict(self, new_frame):
        '''
        Given frame object that is newly placed, update inventory dictionary 
        Input
            TrussFrameRL object
        Updates
            self.inventory_dict : dictionary of inventory of free frame types : inventory count
        '''
        if self.inventory_dict[new_frame.type_structure] == -1: # infinite inventory
            pass
        else :
            self.inventory_dict[new_frame.type_structure] -= 1
            # print(f'(update_inventory_dict) used {new_frame.type_structure} : {self.inventory_dict[new_frame.type_structure]}')

    def update_frame_grid(self, new_frame):
        '''
        Given new frame object, update current frame grid where 
        Input
            TrussFrameRL object 
        Updates:
        - self.curr_frame_grid : A grid where each cell is updated based on the frame type.
            - (cell state for array size frame_grid_size_x frame_grid_size_y) 
            - grid where each cell has value unoccupied= 0, free frame (light) = 2, support frame = 1, force = -1
        - self.valid_pos : A set of valid (x, y) frame positions on the frame grid where a new frame can be placed.
        '''
        # Update the current frame grid with the new frame's type
        self.curr_frame_grid[new_frame.x_frame, new_frame.y_frame] = new_frame.type_structure.idx

        # Remove the position of the new frame from valid_pos if it exists
        if (new_frame.x_frame, new_frame.y_frame) in self.valid_pos:
            self.valid_pos.remove((new_frame.x_frame, new_frame.y_frame))
        # else:
        #     raise ValueError(f"Position ({new_frame.x_frame}, {new_frame.y_frame}) is not a valid position for placing a frame.")

        # Update valid position if frame not load frame
        if new_frame.type_structure != FrameStructureType.EXTERNAL_FORCE: 
            # Check the four adjacent cells within the frame grid 
            adj_cells = [
                (new_frame.x_frame - 1, new_frame.y_frame),  # Left
                (new_frame.x_frame + 1, new_frame.y_frame),  # Right
                (new_frame.x_frame, new_frame.y_frame - 1),  # Below
                (new_frame.x_frame, new_frame.y_frame + 1)   # Above
            ]

            # Add adjacent cells to valid_pos if they are unoccupied (value 0)
            for (x_adj, y_adj) in adj_cells:
                # Check if the adjacent cell is within the frame grid bounds
                if 0 <= x_adj < self.frame_grid_size_x and 0 <= y_adj < self.frame_grid_size_y:
                    # print(f'self.frame_grid_size_x, frame_grid_size_y : {self.frame_grid_size_x, self.frame_grid_size_y}')
                    # If the adjacent cell is unoccupied or load placeholder frame, add it to valid_pos
                    if self.curr_frame_grid[x_adj, y_adj] == FrameStructureType.UNOCCUPIED.idx \
                                                                or self.curr_frame_grid[x_adj, y_adj] == FrameStructureType.EXTERNAL_FORCE.idx:
                        if (x_adj, y_adj) not in self.valid_pos:
                            self.valid_pos.add((x_adj, y_adj)) # 

    def update_frame_graph(self, new_frame):
        '''
        Given new TrussFrameRL object that is placed, update current frame graph where 
            - nodes are TrussFrameRL objects
            - edges are physical adjacencies, relative distance to external forces and supports
        '''
        pass #TODO
    
    def update_fea_graph(self, new_frame, t_load_mag=[0.0, 0.0, 0.0]):
        '''
        Input 
            new_frame : TrussFrameRL object (centroid, frame type)

        Given new TrussFrameRL object, update self.curr_feagraph (FEAgraph object) where 
            self.vertices : dictionary of vertices with coordinate as key and Vertex object as value
            self.edges : adjacency list of tuples of vertex indices pairs
            self.maximal_edges : dictionary where key is direction and value is list of MaximalEdge objects 
            {
                'horizontal': [],
                'vertical': [],
                'LB_RT': [],
                'LT_RB': []
            }
            self.load : A list of tuples (node.id, [load.x, load.y, load.z]) 

        Update current FEAgraph with added truss frame so that existing node indices are preserved
        1. merge overlapping new nodes with existing nodes
        2. check line overlap with existing edge using maximal edge representation 
        3. update edge list with new line segments

        '''
        # Calculate the positions of the four vertices
        half_size = self.frame_size // 2
        vert_pos = [
            (new_frame.x - half_size, new_frame.y - half_size),  # Bottom-left
            (new_frame.x + half_size, new_frame.y - half_size),  # Bottom-right
            (new_frame.x + half_size, new_frame.y + half_size),  # Top-right
            (new_frame.x - half_size, new_frame.y + half_size)   # Top-left
        ]

        if new_frame.type_structure == FrameStructureType.EXTERNAL_FORCE: # target frame
            # Bottom right vertex of proxy frame is registered as load but not as Vertex object
            target_load_pos = vert_pos[1]
            self.curr_fea_graph.external_loads[target_load_pos] = t_load_mag
            # TODO need to cross check with existing Vertices if target load is added after environment init
        else: # free or support
            new_vertices = [] # Vertex object in order of bottom-left, bottom-right, top-right, top-left
            for i, pos in enumerate(vert_pos):
                # If new node overlaps with existing node, merge (preserve existing node attributes - id, is_free)
                if pos in self.curr_fea_graph.vertices:
                    new_v = self.curr_fea_graph.vertices[pos] # get overlapping existing node
                    # allow change free->fixed but not fixed->free
                    if new_frame.type_structure == FrameStructureType.SUPPORT_FRAME:
                        new_v.is_free = False
                else: # If node does not overlap with existing node, create new node 
                    is_free = None
                    if new_frame.type_structure == FrameStructureType.LIGHT_FREE_FRAME or new_frame.type_structure == FrameStructureType.MEDIUM_FREE_FRAME: # Free 
                        # new_v = Vertex(pos, is_free=True, load=new_frame.type_structure.node_load)
                        is_free = True
                    elif new_frame.type_structure == FrameStructureType.SUPPORT_FRAME: # Support
                        if i==0 or i==1: # Bottom left, Bottom right are fixed 
                            # new_v = Vertex(pos, is_free=False, load=new_frame.type_structure.node_load)
                            is_free = False
                            self.curr_fea_graph.supports.append(pos) # add to list of supports
                        else: # Top left, Top right are free
                            is_free = True
                            # new_v = Vertex(pos, is_free=True, load=new_frame.type_structure.node_load)
                    new_v = Vertex(pos, is_free=is_free, load=new_frame.type_structure.node_load)
                    
                    # additionally check if meets with external load
                    if pos in self.curr_fea_graph.external_loads:
                        # new_v.load += self.curr_fea_graph.external_loads[pos]
                        new_v.load = [x + y for x, y in zip(new_v.load, self.curr_fea_graph.external_loads[pos])]

                    # add new node to fea graph
                    self.curr_fea_graph.vertices[pos] = new_v 
                # add to new vertices to combine edges                    
                new_vertices.append(new_v) 

            # Check line overlap with existing edge using maximal edge representation 
            self.curr_fea_graph.combine_and_merge_edges(frame_type_shape=new_frame.type_shape, new_vertices=new_vertices)

    def update_displacement(self):
        '''
        Used in apply_action 
        Called upon action that indicates end of episode (end_bool == 1)
        Performs FEA with Julia ASAP 
        Updates self.curr_fea_graph displacement attribute
        '''
        # print('Updating Displacement...')

        # Solve truss model with ASAP
        module_name = f"TrussFrameRL_{self.env_idx}"
        jl = juliacall.newmodule(module_name) 
        # jl = juliacall.newmodule("TrussFrameRL") 
        curr_env = jl.seval('Base.active_project()')
        # print(f"The current active Julia environment is located at: {curr_env}")

        # Step 0: Initialize Julia session and import necessary Julia modules
        # jl.seval('using AsapToolkit')
        jl.seval('using Asap')

        truss_analysis_path = os.path.join(PARENT_DIR, "..", "TrussFrameMechanics", "truss_analysis.jl")
        # Include the Julia file using the absolute path
        jl.include(truss_analysis_path)
        # jl.include("TrussFrameMechanics/truss_analysis.jl") # system path error
        jl.seval('using .TrussAnalysis')

        displacement, failed_elements = pythonAsap.solve_fea(jl, self.curr_fea_graph, self.frame_length_m) # return nodal displacement
        self.curr_fea_graph.displacement = displacement
        self.curr_fea_graph.failed_elements = failed_elements

    def update_target_meet(self, new_frame):
        '''
        Used in apply_action
        Input
            new_frame 
        Check if target loads are met (top right or top left node of frame meets with external load node) 
        Update self.target_loads_met and self.is_connected in place
        '''
        top_right = (new_frame.x + self.frame_size//2, new_frame.y + self.frame_size//2)
        top_left = (new_frame.x - self.frame_size//2, new_frame.y + self.frame_size//2)
        for target in self.target_loads_met:
            if top_right == target:
                self.target_loads_met[target] = True
            if top_left == target:
                self.target_loads_met[target] = True
        if all(self.target_loads_met.values()):
            self.is_connected = True

    ## Drawing
    def draw_truss_analysis(self):
        '''
        Used within take step after episode as ended with connection
        Given that displacement has been updated
        Overlay displaced truss to plot by updating self.fig and self.ax based on self.curr_fea_graph.displacement
        Overlay failed elements in red based on self.curr_fea_graph.failed_elements
        '''
        displaced_truss_color = 'gray'
        disp_vis_scale = 10 # scale displacement for visualization 
        
        # Get Displaced vertices
        displaced_vertices = {} # node id : (x, y)
        max_disp = None # (node_id, displacement magnitude) 
        for i, (coord, V) in enumerate(self.curr_fea_graph.vertices.items()):
            dx, dy = self.curr_fea_graph.displacement[i][:2] # output 2D list of nodal displacement [x,y,z] for each node in node index order. Only non-empty upon fea (eps end)
            # Calculate the displacement magnitude
            # d_mag = np.sqrt((dx/ disp_vis_scale)**2 + (dy/ disp_vis_scale)**2) # Scale down if necessary for visualization
            d_mag = np.sqrt(dx**2 + dy**2)
            if max_disp == None or d_mag >= max_disp[1]:
                max_disp = (V.id, d_mag) 
            # print(f'displacement for node {i} is {dx}, {dy}')
            new_x = coord[0] + dx *  disp_vis_scale
            new_y = coord[1] + dy *  disp_vis_scale

            displaced_vertices[V.id] = (new_x, new_y)
            self.ax.add_patch(patches.Circle((new_x, new_y), radius=0.05, color=displaced_truss_color, alpha=0.8))
            # Add text showing displacement magnitude next to each circle
            self.ax.text(new_x + 0.1, new_y + 0.1, f'{d_mag:.2f}', color='gray', fontsize=8)
        
        # Connect deflected nodes with edges
        for edge in self.curr_fea_graph.edges:
            start_id, end_id = edge  # node ids
            start_coord = displaced_vertices[start_id]
            end_coord = displaced_vertices[end_id]

            # Plot the deflected truss member
            self.ax.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]],
                    color=displaced_truss_color, linestyle='--', linewidth=1)
        for edge in self.curr_fea_graph.failed_elements:
            start_id, end_id = edge
            start_coord = displaced_vertices[start_id]
            end_coord = displaced_vertices[end_id]
            self.ax.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]],
                    color='red', linestyle='-', linewidth=3)
        # Highlight max displacement
        # Find the maximum displacement index and value
        # max_disp_index = np.argmax([np.linalg.norm(d[:2]) for d in self.curr_fea_graph.displacement])
        # max_disp_value = np.linalg.norm(self.curr_fea_graph.displacement[max_disp_index][:2])
        # max_disp_coord = displaced_vertices[max_disp_index]
        # Add text to highlight the max displacement
        maxd_x_coord, maxd_y_coord = displaced_vertices[max_disp[0]]
        maxd_value = max_disp[1]
        self.max_deflection = max_disp[1]
        
        # Overlay max deflection value on the max deflection node
        # if self.max_deflection >= self.allowable_deflection:
        #     self.ax.text(maxd_x_coord+0.1, maxd_y_coord+0.2, f'{maxd_value:.5f}', color='red', fontsize=11)
        # else:
        #     self.ax.text(maxd_x_coord+0.1, maxd_y_coord+0.2, f'{maxd_value:.5f}', color='green', fontsize=11)
        # Draw circle around max delfected node 
        self.ax.add_patch(patches.Circle((new_x, new_y), radius=0.2, color='red', alpha=0.5))

    def draw_fea_graph(self):
        '''
        Update self.fig and self.ax based on self.curr_fea_graph
        used in render_frame
        '''
        text_offset = 0.1

        vertices = self.curr_fea_graph.vertices.items() # coord, Vertex object pairs
        # maximal_edges = self.curr_fea_graph.maximal_edges.items()
        # supports = self.curr_fea_graph.supports # list of board coords where the nodes are supports / pinned (as opposed to free)
        # Draw vertices
        for coord, vertex in vertices:
            # if vertex.coordinates in supports:
            if vertex.is_free == False:
                # self.ax.add_patch(patches.Rectangle((coord[0] - 0.1, coord[1] - 0.1), 0.2, 0.2, color='blue', lw=1.5, fill=True))
                # Create a triangle with the top point at the vertex coordinate
                triangle_vertices = [
                    (coord[0], coord[1]),  # Top point
                    (coord[0] - 0.15, coord[1] - 0.2),  # Bottom-left point
                    (coord[0] + 0.15, coord[1] - 0.2)   # Bottom-right point
                ]
                self.ax.add_patch(patches.Polygon(triangle_vertices, color='blue', lw=1.5, fill=True))
            else:
                self.ax.add_patch(patches.Circle(coord, radius=0.1, color='blue', lw=1.5, fill=False ))
            self.ax.text(coord[0]-text_offset, coord[1]+text_offset, 
                         str(vertex.id), 
                         fontsize=10, ha='right', color='black')
            
        # Draw frame edges
        for trussframe in self.frames:
            # outer frame
            if trussframe.type_structure == FrameStructureType.SUPPORT_FRAME or trussframe.type_structure == FrameStructureType.LIGHT_FREE_FRAME:
                self.ax.add_patch(patches.Rectangle((trussframe.x - self.frame_size//2, trussframe.y - self.frame_size//2), self.frame_size, self.frame_size, color='black', lw=1.5, fill=False))
            elif trussframe.type_structure == FrameStructureType.MEDIUM_FREE_FRAME:
                self.ax.add_patch(patches.Rectangle((trussframe.x - self.frame_size//2, trussframe.y - self.frame_size//2), self.frame_size, self.frame_size, facecolor=((0.7, 0.7, 0.7, 0.8)), edgecolor = 'black', lw=1.5, fill=True))
            # brace
            if trussframe.type_shape == FrameShapeType.DOUBLE_DIAGONAL:
                # Add diagonal lines from left bottom to right top, right bottom to left top
                # Coordinates of the rectangle corners
                x0 = trussframe.x - self.frame_size / 2
                y0 = trussframe.y - self.frame_size / 2
                x1 = trussframe.x + self.frame_size / 2
                y1 = trussframe.y + self.frame_size / 2
                # Diagonal from bottom-left to top-right
                line1 = mlines.Line2D([x0, x1], [y0, y1], color='black', lw=1)
                self.ax.add_line(line1)
                # Diagonal from top-left to bottom-right
                line2 = mlines.Line2D([x0, x1], [y1, y0], color='black', lw=1)
                self.ax.add_line(line2)


        # # Draw maximal edges (optional, if visually distinct from normal edges)
        # for direction, edges in maximal_edges:
        #     for edge in edges:
        #         if len(edge.vertices) >= 2:
        #             # Get start and end vertices from the list of vertices
        #             start_me = edge.vertices[0]
        #             end_me = edge.vertices[-1]
                    
        #             # Draw the line connecting the start and end vertices
        #             self.ax.plot([start_me.coordinates[0], end_me.coordinates[0]], 
        #                         [start_me.coordinates[1], end_me.coordinates[1]], 
        #                         color='black', linestyle='-', linewidth=1)
        #         else:
        #             print(f"Warning: Maximal edge in direction {direction} has less than 2 vertices and cannot be drawn.")
        
        # Draw external forces as red arrows
        for coord, load in self.curr_fea_graph.external_loads.items():
            force_magnitude = (load[0]**2 + load[1]**2 + load[2]**2)**0.5
            if force_magnitude > 0:
                arrow_dx = load[0] * 0.01
                arrow_dy = load[1] * 0.01
                arrow_tail_x = coord[0] - arrow_dx
                arrow_tail_y = coord[1] - arrow_dy
                # arrow_head_x = arrow_tail_x - arrow_dx
                # arrow_head_y = arrow_tail_y - arrow_dy

                self.ax.arrow(arrow_tail_x, arrow_tail_y, arrow_dx, arrow_dy+0.1, head_width=0.2, head_length=0.2, fc='red', ec='red', linewidth=1.5)
                self.ax.text(arrow_tail_x, arrow_tail_y + 0.1, f"{force_magnitude:.2f} kN", color='red', fontsize=12)
    
    # Render 
    def render_frame(self):
        '''
        initialize and updates self.ax, self.fig object 
        '''
        # Create the figure and axes
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        # self.ax.set_xlim([0, self.board_size_x])
        # self.ax.set_ylim([0, self.board_size_y])
        margin = 1
        self.ax.set_xlim([-margin, self.board_size_x + margin])
        self.ax.set_ylim([-margin, self.board_size_y + margin])
        self.ax.set_aspect('equal', adjustable='box')
        # self.ax.set_xticks(range(self.board_size_x + 1))
        # self.ax.set_yticks(range(self.board_size_y + 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        # Draw grid lines
        # self.ax.grid(True, which='both', color='lightblue', linestyle='-', linewidth=0.5,  zorder=0)
        for i in range(0, self.board_size_x + 1, 2):
            self.ax.axvline(x=i, color='lightblue', linestyle='-', linewidth=2, zorder=0)
        for j in range(0, self.board_size_y + 1, 2):
            self.ax.axhline(y=j, color='lightblue', linestyle='-', linewidth=2, zorder=0)

        # Highlight valid position cells (except for final frame if terminated)
        if self.eps_end_valid == False:
            for frame_x, frame_y in self.valid_pos:
                x , y = self.framegrid_to_board(frame_x, frame_y)
                rect = patches.Rectangle((x - self.frame_size//2, y - self.frame_size//2), 
                                        self.frame_size, self.frame_size, 
                                        linewidth=0, edgecolor='lightblue', facecolor='lightblue')
                self.ax.add_patch(rect)
            
        # Draw current fea graph
        self.draw_fea_graph() # update self.fig, self.ax with current fea graph 

        # Overlay with displacement graph
        if self.eps_end_valid == True:
        # if len(self.curr_fea_graph.displacement) != 0 : # check if displacement has been analyzed 
            self.draw_truss_analysis() # last plot has displaced structure 
            # self.ax.text(
            #                 0.5, -0.05,  # x=0.5 centers the text, y=0.01 places it at the bottom
            #                 # f'Allowable Deflection : {self.allowable_deflection:.4f} m',
            #                 f'Allowable Deflection : {self.allowable_deflection:.3f} m, Max Deflection: {self.max_deflection:.3f} m, Reward: {self.episode_return:.1f}, Episode Length: {self.episode_length}',
            #                 color='black',
            #                 fontsize=14,
            #                 ha='center',  # Center-aligns the text horizontally
            #                 transform=self.ax.transAxes  # Use axis coordinates
            #             )

            # Caption at termination
            caption_fontsize_large = 16
            caption_fontsize_small = 12
            self.ax.text(
                0.1, -0.05,  # Adjust x to position the text correctly
                'Allowable Deflection :',
                color='black',
                fontsize=caption_fontsize_small,
                ha='center',  # Center-aligns the text horizontally
                transform=self.ax.transAxes  # Use axis coordinates
            )

            self.ax.text(
                0.25, -0.05,  # Adjust x to position the value correctly
                f'{self.allowable_deflection:.3f} m',
                color='black',
                fontsize=caption_fontsize_small,
                ha='center',  # Center-aligns the text horizontally
                transform=self.ax.transAxes  # Use axis coordinates
            )

            self.ax.text(
                0.4, -0.05,  # Adjust x to position the text correctly
                'Max Deflection:',
                color='black',
                fontsize=caption_fontsize_small,
                ha='center',  # Center-aligns the text horizontally
                transform=self.ax.transAxes  # Use axis coordinates
            )

            if self.max_deflection >= self.allowable_deflection:
                self.ax.text(
                    0.5, -0.05,  # Adjust x to position the value correctly
                    f'{self.max_deflection:.3f} m',
                    color='red',
                    fontsize=caption_fontsize_large,
                    ha='center',  # Center-aligns the text horizontally
                    transform=self.ax.transAxes  # Use axis coordinates
                )
            else:
                self.ax.text(
                    0.525, -0.05,  # Adjust x to position the value correctly
                    f'{self.max_deflection:.3f} m',
                    color='green',
                    fontsize=caption_fontsize_large,
                    ha='center',  # Center-aligns the text horizontally
                    transform=self.ax.transAxes  # Use axis coordinates
                )

            self.ax.text(
                0.625, -0.05,  # Adjust x to position the text correctly
                'Reward:',
                color='black',
                fontsize=caption_fontsize_small,
                ha='center',  # Center-aligns the text horizontally
                transform=self.ax.transAxes  # Use axis coordinates
            )

            self.ax.text(
                0.7, -0.05,  # Adjust x to position the value correctly
                f'{self.episode_return:.1f}',
                color='green',
                fontsize=caption_fontsize_large,
                ha='center',  # Center-aligns the text horizontally
                transform=self.ax.transAxes  # Use axis coordinates
            )

            self.ax.text(
                0.825, -0.05,  # Adjust x to position the text correctly
                'Episode Length:',
                color='black',
                fontsize=caption_fontsize_small,
                ha='center',  # Center-aligns the text horizontally
                transform=self.ax.transAxes  # Use axis coordinates
            )

            self.ax.text(
                0.95, -0.05,  # Adjust x to position the value correctly
                f'{self.episode_length}',
                color='black',
                fontsize=caption_fontsize_small,
                ha='center',  # Center-aligns the text horizontally
                transform=self.ax.transAxes  # Use axis coordinates
            )
        else:
            pass
            # print(f'Displacement is empty!')

        # # Interactive (debug_all Mode)
        # if self.render_mode == 'debug_all':
        #     # Ensure the canvas is available
        #     self.fig.canvas.draw()
        #     self.click_event_id = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def render(self):
        ''' 
        Used in step() to generate plot of valid action (rgb_list) or any action (debug_all)
        Render Modes
            'debug_all' : Used in step to plot every step 
            'rgb_list' : Used in render intermediary between render_frame and render to convert plot to rgb array  
        '''
        if self.render_mode == "rgb_list":
            if self.render_valid_action: # only save valid action frames
                self.render_frame() # initialize and update self.ax, self.fig object
                img = self.fig_to_rgb_array(self.fig)
                self.render_list.append(img)
                # plt.show() # DEBUG
                plt.close(self.fig)
                # print(f'returning img for rgb_array!')
        elif self.render_mode == "rgb_end":
            render_name = f"end_{self.render_counter}.png" 
            if self.eps_end_valid:
                self.render_frame()
                render_path = os.path.join(self.render_dir, render_name)
                # Save the render
                self.render_frame()
                plt.savefig(render_path, bbox_inches='tight')
                # Increment the counter for the next file
                self.render_counter += 1
        elif self.render_mode == "debug_valid":
            if self.render_valid_action:
                self.render_frame()
                plt.show()
                plt.close(self.fig)
        elif self.render_mode == "debug_end":
            if self.eps_end_valid:
                self.render_frame()
                plt.show()
                plt.close(self.fig)
        elif self.render_mode == "debug_all":
            self.render_frame() # initialize and update self.ax, self.fig object
            plt.show()
            plt.close(self.fig)

        elif self.render_mode == None:
            pass
        
        else:
            raise NotImplementedError(f"Render mode {self.render_mode} is not supported")
        
    def get_render_list(self):
        '''
        Called within gymnasium.utils.save_video function
            - 'rgb_list' : return list of fig in rgb format to be saved in RenderList Wrapper item 
                            Given that we are working with 
        '''
        if self.render_mode == "rgb_list":
            return self.render_list # list of fig objects
        
        else:
            raise NotImplementedError(f"Render mode {self.render_mode} is not supported")
    
    def fig_to_rgb_array(self, fig):
        '''
        Draw the figure on the canvas
        Used to save plt within render_list
        '''
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba() # Retrieve the image as a string buffer
        X = np.asarray(buf) # Convert to a NumPy array
        rgb_array = X[:, :, :3] # Convert RGBA to RGB
        
        return rgb_array

    # Utils
    def framegrid_to_board(self, frame_x, frame_y):
        '''
        Input
            (frame_x, frame_y) coordinates within frame grid
        Output
            board_coords : (x,y) centroid board coords of frame 
        '''
        board_x = frame_x*self.frame_size + self.frame_size//2
        board_y = frame_y*self.frame_size + self.frame_size//2
        return (board_x, board_y)
    

    # # Interactive (debug_all mode)
    # def get_cursor_location(self, event):
    #     """
    #     Get user cursor location within the canvas, translate that to frame grid coordinates within the environment.
    #     """
    #     # Get the cursor location within the canvas
    #     canvas_x, canvas_y = event.x, event.y

    #     # Translate canvas coordinates to frame grid coordinates
    #     inv = self.fig.transFigure.inverted()
    #     fig_x, fig_y = inv.transform((canvas_x, canvas_y))

    #     # Assuming the canvas coordinates are normalized (0 to 1)
    #     frame_x = int(fig_x * self.frame_grid_size_x)
    #     frame_y = int(fig_y * self.frame_grid_size_y - 1)

    #     return frame_x, frame_y

    # def on_click(self, event):
    #     frame_x, frame_y = self.get_cursor_location(event)
    #     self.click_frame_x = frame_x
    #     self.click_frame_y = frame_y
    #     print(f"Frame grid coordinates: ({frame_x}, {frame_y})")
    
    # Debugging
    def print_framegrid(self):
        '''
        Prints the current frame grid in a human-readable format with the x-axis across different lines.
        '''
        print("Current Frame Grid:")
        transposed_grid = self.curr_frame_grid.T  # Transpose the grid
        for row in reversed(transposed_grid):
            print(" ".join(map(str, row)))


    def get_action_mask(self):
        """
        Used to generate action mask based on current state
        action : (end_bool, freeframe_idx, frame_x, frame_y)
            Absolute Coordinates (end_bool, freeframe_idx, frame_x, frame_y)
                End bool : 0,1 (False, True)
                Free Frame index : 0, 1  (light, medium) (FrameStructureType index - 2)
                Frame x, y : 0 to self.frame_grid_size_x, self.frame_grid_size_y
        invalid action is defined as union of following cases 
            - frame_x, frame_y not in valid position (tangent cells to existing design)
            - freeframe_idx inventory is 0 (other frame types are not used up)
            - end_bool = True but support and target loads are not connected
        in other words valid actions are defined as intersection of following cases
            - frame_x, frame_y in valid position (tangent cells to existing design)
            - freeframe_idx inventory is not 0 (other frame types are not used up)
            - end_bool = False if support and target loads are connected and True/False if not connectedif not connected
        action mask is used in rollout as env.sample(mask=curr_mask)
        """
        # initialize action mask using self.action_space 
        action_mask = np.zeros(self.action_space.n, dtype=np.int8)
        # Get all raw valid action vectors based on current state (end_bool, freeframe_idx, frame_x, frame_y) using self.valid_pos and self.inventory_dict, self.is_connected
        valid_actions = []
        for freeframe_idx in [0, 1]:
            freeframe_type = FrameStructureType.get_framestructuretype_from_idx(freeframe_idx+2)# dictionary key is FrameStructureType
            for frame_x, frame_y in self.valid_pos:
                if self.inventory_dict[freeframe_type] > 0:
                    # forward look ahead to check if support and target loads are connected
                    temp_is_connected = self.check_is_connected(frame_x, frame_y)
                    if temp_is_connected == True:
                        for end_bool in [False, True]:
                            valid_actions.append((end_bool, freeframe_idx, frame_x, frame_y))
                    else:
                        end_bool = False
                        valid_actions.append((end_bool, freeframe_idx, frame_x, frame_y))
        
        # encode action vectors to action integers using self.action_converter.encode(action)
        valid_action_ints = [self.action_converter.encode(action) for action in valid_actions]
        # print(f'valid action idx : {valid_action_ints}')
        
        # apply 1 to valid actions and 0 to invalid actions
        action_mask[valid_action_ints] = 1
        # action_mask[np.logical_not(action_mask)] = 0
        
        return action_mask