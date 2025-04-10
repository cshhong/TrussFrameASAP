'''

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
print(f'Parent Directory of cantilever env v0 added to sys.path : {PARENT_DIR}')

# Add the TrussFrameMechanics directory to sys.path
TRUSS_FRAME_ASAP_PATH = os.path.abspath(os.path.join(PARENT_DIR, '..'))
sys.path.append(TRUSS_FRAME_ASAP_PATH)
print(f'TrussFrameMechanics path added to sys.path: {TRUSS_FRAME_ASAP_PATH}')

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
                - grid where each cell has value 
                    force = -1, (one cell per 100kN)
                    unoccupied= 0, 
                    support frame = 1,
                    free frame (light) = 2,  
                    free frame (medium) = 3,
                    inventory light = 4,
                    inventory medium = 5,
                    
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
            
            (obs_mode)'frame_grid_singleint' : single int encoding of frame_grid with inventory values
            (obs_mode)'frame_grid' : frame_grid representation + added row with inventory values at end of row

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
                    failed_elements : A list of element index pairs that failed with compression / tension info         (node_idx1, node_idx2, compression-0/tension-1)
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
    metadata = {"render_modes": [None, "debug_all", "debug_valid", "rgb_list", "debug_end", "rgb_end", "rgb_end_interval", "human_playable"], 
                "render_fps": 1,
                "obs_modes" : ['frame_grid_singleint', 'frame_grid', 'fea_graph', 'frame_graph'],
                }


    def __init__(self,
                 render_mode = None,
                 frame_grid_size_x=10,
                 frame_grid_size_y=5,
                 frame_size=2, 
                 render_interval_eps=500,
                 render_interval_consecutive=5,
                 render_dir = 'render',
                 max_episode_length = 40,
                 obs_mode='frame_grid_singleint',
                 env_idx = 0,
                 rand_init_seed = None,
                 bc_height_options=[1,2],
                 bc_length_options=[3,4,5],
                 bc_loadmag_options=[300,400,500],
                 bc_inventory_options=[(10,10), (10,5), (5,5), (8,3)],
                 ):
        # super().__init__()

        # Calculate the size of the frame grid based on the frame size
        self.frame_grid_size_x = frame_grid_size_x
        self.frame_grid_size_y = frame_grid_size_y
        self.frame_size = frame_size # visualization grid coordinates (*physical frame length in m is sized by frame_length_m used in truss_analysis.jl)
        self.board_size_x = self.frame_size * self.frame_grid_size_x # visualized board size
        self.board_size_y = self.frame_size * self.frame_grid_size_y # visualized board size

        self.max_episode_length = max_episode_length
        self.global_steps = 0 # activated taking valid action in main (not env.step!)
        self.global_terminated_episodes = 0
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
        if self.render_mode == "rgb_end" or self.render_mode == "rgb_end_interval":
            self.render_counter = 0
        self.render_interval_eps = render_interval_eps
        self.render_interval_consecutive = render_interval_consecutive # number of episodes to render consecutively at interval

        # Current State (Frames)
        self.frames=[] # stores TrussFrameRL objects in sequence of creation
        # self.support_frames = [] # list of TrussFrameRL objects
        

        # Boundary Conditions
        self.frame_length_m = 3.0 # actual length of frame in meters used in truss analysis
        # from args
        self.bc_height_options = bc_height_options
        self.bc_length_options = bc_length_options
        self.bc_loadmag_options = bc_loadmag_options
        self.bc_inventory_options = bc_inventory_options
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
        self.is_connected_fraction = 0 # fraction of target loads connected to support

        self.eps_terminate_valid = False # after applying action with end_bool=1, signify is episode end is valid, used in visualization

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
        self.bc_inventory = None # dictionary of inventory in order of free frame types this does not change after reset
        self.inventory_dict = None # dictionary of inventory in order of free frame types
        #     inventory = {
        #     FrameStructureType.LIGHT_FREE_FRAME : light_inv, # -1 indicate no limits
        #     FrameStructureType.MEDIUM_FREE_FRAME : med_inv,
        #     # FrameStructureType.HEAVY_FREE_FRAME : *,
        # }
        self.n_all_inventory = None # total number of inventory
        self.obs_converter = None # ObservationBijectiveMapping object, used to encode and decode observations
        self.action_converter = None # ActionBijectiveMapping object, used to encode and decode actions
        self.observation_space = None
        self.single_observation_space = None
        self.action_space = None
        self.single_action_space = None

        # Random Initialization
        self.n_rand_init_steps = 0 # initialized in training function
        self.rand_init_actions = [] # random action int appended in training function
        self.rand_init_seed = rand_init_seed # seed for random action initialization
        self.reset_env_bool = False # boolean to indicate if env is reset (before taking first action) to initialize random actions

        # Set boundary conditions ; support, target load, inventory
        self.initBoundaryConditions() # set self.allowable_deflection, self.inventory_dict, self.frames, self.curr_frame_grid, self.target_loads_met, FrameStructureType.EXTERNAL_FORCE.node_load
        self.set_space_converters(self.inventory_dict) # set self.obs_converter, self.action_converter
        self.set_gym_spaces(self.obs_converter, self.action_converter) # set self.observation_space, self.action_space

        self.bc_condition = None # condition of boundary conditions (height, length, loadmag, inventory_light, inventory_med) used to condition actor, critic network

        # Human Playable mode 
        if self.render_mode == "human_playable":
            self.click_event_id = None # created in render_frame
            self.key_event_id = None # created in render_frame
            
            # # Connect the button press event (add frame)
            # self.click_event_id = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            # # Connect the keypress event (select frame type)
            # self.key_event_id = self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
            
            self.human_action_frame_type = None # type of frame type selected by human through key press : 1 for FreeFrameType.LIGHT_FREE_FRAME, 2 for FreeFrameType.MEDIUM_FREE_FRAME
            self.human_action_end = False # boolean to end episode selected by human through key press of 'e' : end episode, 'c' : continue episode
            self.human_action_frame_coords = None # coordinates of frame selected by human through click (frame_x, frame_y)

        print("Initialized Cantilever Env!")
        # self.render()


    def reset(self, seed=None, **kwargs):
        '''
        Create boundary condition within environment with 
            generate_bc.set_cantilever_env_framegrid(self.frame_grid_size_x)
        that returns support_frames, targetload_frames within the frame grid
        self.frames, self.valid_pos, self.curr_frame_grid, self.curr_frame_graph, self.curr_fea_graph is updated

        '''
        # print('Resetting Env!')

        self.render_list = []
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
        
        self.eps_terminate_valid = False

        self.render_valid_action = True # temporarily turn on to trigger render
        self.render()
        self.render_valid_action = False
        # print(f"valid pos : {self.valid_pos}")

        self.rand_init_actions = [] # reset random init actions
        self.reset_env_bool = True # set to True to initialize random actions in training function

        inventory_array = np.array(list(self.inventory_dict.values()), dtype=np.int64)
        if self.obs_mode == 'frame_grid_singleint':
            obs = self.obs_converter.encode(self.curr_frame_grid, inventory_array) # encoded int value obs
        elif self.obs_mode == 'frame_grid':
            obs = self.get_frame_grid_observation()
            # print(f'Reset obs : \n{self.curr_frame_grid} \n Inventory : {inventory_array} \n obs : \n{obs}')
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
        support_frames, targetload_frames, inventory_dict, cantilever_length_f = \
            generate_bc.set_cantilever_env_framegrid(self.frame_grid_size_x,
                                                    height_options = self.bc_height_options,
                                                    length_options = self.bc_length_options,
                                                    magnitude_options = self.bc_loadmag_options,
                                                    inventory_options = self.bc_inventory_options,)
        self.cantilever_length_f = cantilever_length_f
        self.allowable_deflection = self.frame_length_m * cantilever_length_f / 120 # length of cantilever(m) / 120
        # self.extr_load_mag = list(targetload_frames.values())[0] # magnitude of external load in kN (x,y,z)
        self.inventory_dict = inventory_dict
        self.bc_inventory = inventory_dict.copy() 
        self.n_all_inventory = sum(inventory_dict.values())

        # set FrameStructureType.EXTERNAL_FORCE magnitude values TODO where is this used? 
        #TODO handle multiple target loads
        FrameStructureType.EXTERNAL_FORCE.node_load = list(targetload_frames.values())[0]

        target_condition = [item for (x_frame, y_frame), forces in targetload_frames.items() for item in (x_frame, y_frame, forces[1])] # list of (x_frame, y_frame, y_forcemag) of target loads
        inventory_condition = list(inventory_dict.values()) # list of inventory values
        self.bc_condition = target_condition + inventory_condition

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
            self.update_frame_grid(new_t_frame, t_load_mag=t_load_mag)
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
        if self.obs_mode == "frame_grid_singleint":
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
        
    def set_gym_spaces(self, obs_converter=None, action_converter=None):
        '''
        set self.observation_space, self.action_space according to obs mode
        Assumes that observation and action converters are provided as input
        observation space is continuous (Box) and action space is discrete (Discrete)

        Input
            "frame_grid_singleint" mode 
                obs_converter : ObservationDownsamplingMapping object
                action_converter : ActionBijectiveMapping object
        '''
        # Set observation_space and action_space (following Gymnasium)
        if self.obs_mode == 'frame_grid_singleint':
            assert obs_converter is not None, "obs_converter must be an instance of ObservationBijectiveMapping"
            assert action_converter is not None, "action_converter must be an instance of ActionBijectiveMapping"
            
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
            self.action_space = Discrete(n=n_actions, seed=self.rand_init_seed)
            print(f'Action Space : {self.action_space} sampling seed : {self.rand_init_seed}')
            self.single_action_space = self.action_space

            # Relative Coordinates? (end_bool, frame_idx, left/right/top/bottom) ---> Doesn't save much from absolute coordinates, still have to check valid action!
        
        elif self.obs_mode == 'frame_grid':
            # Define Observations : frame grid with extra row with medium inventory values
            self.observation_space = Box(low=-1, high=4, shape=(self.frame_grid_size_x, self.frame_grid_size_y+2), dtype=np.int64)
            self.single_observation_space = self.observation_space
            print(f'Obs Mode : "{self.obs_mode}" | Obs Space : {self.observation_space} | Single obs space : {self.single_observation_space}')

            # Define Actions : end boolean, freeframe_type, frame_x, frame_y
            n_actions = action_converter._calculate_total_space_size()
            self.action_space = Discrete(n=n_actions, seed=self.rand_init_seed)
            print(f'Action Space : {self.action_space} sampling seed : {self.rand_init_seed}')
            self.single_action_space = self.action_space

        elif self.obs_mode == 'fea_graph':
            print('TODO Need to implement set_gym_spaces for fea_graph!')
            pass
            # Gymnasium Composite Spaces - Graph 
            # Graph - node_features (frame type), edge_features(connection direction), edge_links
            # Dict (not directly used in learning but can store human interpretable info)
        elif self.obs_mode == 'frame_graph':
            print('TODO Need to implement set_gym_spaces for frame_graph!')
            pass
            self.observation_space = Graph(node_space=Box(low=0, high=self.n_all_inventory+len(self.target_loads_met), shape=(1,)), edge_space=Discrete(3), seed=1)
    

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
        if self.reset_env_bool==True:
                self.reset_env_bool = False # set to True to initialize random actions in training function

        self.global_steps += 1
        
        self.episode_length += 1
        self.render_valid_action = False # used to trigger render
        reward = 0
        terminated = False
        truncated = False
        info = {}
        action_tuple = self.action_converter.decode(action) # int to action tuple
        end, freeframe_idx, frame_x, frame_y = action_tuple
        end_bool = True if end==1 else False # end is only possible when support and target loads are connected
        
        # Apply action
        # print(f'    applying action : {action_tuple}')
        self.apply_action(action_tuple) # if end updates displacement and failed elements in curr_fea_graph

        # Using masked actions : all actions taken are valid actions 
        # if inventory is used up, episode is truncated
        if all(value <= 0 for value in self.inventory_dict.values()):
            truncated = True # truncated because inventory is empty
        
        # terminate if action ends episode
        if end_bool==True:
            terminated = True
            # print('Episode Terminated!')
            # print(f'    External Load : {self.extr_load_mag}')
            self.eps_terminate_valid = True # used in render_frame to trigger displacement vis, in render to save final img
            self.global_terminated_episodes += 1

            reward += 3 # completion reward (long horizon)

            if self.max_deflection < self.allowable_deflection:
                reward += 3 * self.allowable_deflection / self.max_deflection  # large reward for low deflection e.g. 0.5 / 0.01 = 50, scale for allowable displacement considering varying bc 
                # reward += self.allowable_deflection / self.max_deflection  # large reward for low deflection e.g. 0.5 / 0.01 = 50, scale for allowable displacement considering varying bc 
                # print(f"    Max Deflection : {self.max_deflection} Deflection Reward : {reward}")
                # scale reward according to load 
                # print(f'    deflection reward : {reward}')
                # reward *= abs(self.extr_load_mag[1])/200 # scale reward by external load magnitude in y direction
            
            # reward -= 2*len(self.curr_fea_graph.failed_elements) # large penalty by number of failed elements 
            # reward -= len(self.curr_fea_graph.failed_elements) # large penalty by number of failed elements 
            reward -= len(self.curr_fea_graph.failed_elements) * (1/self.cantilever_length_f) # large penalty by number of failed elements 
            # print(f'    failed penalty added : {reward}')
            # store reward value for render 
        
        
        if truncated and not terminated:
            # reward -= 20 # large penalty for not finishing within inventory 
            # reward -= 2 # large penalty for not finishing within inventory 
            reward = 0

        # Render frame
        self.render_valid_action = True

        # Store trajectory
        inventory_array = np.array(list(self.inventory_dict.values())) # convert inventory dictionary to array in order of free frame types
        if self.obs_mode == 'frame_grid_singleint':
            obs = self.obs_converter.encode(self.curr_frame_grid, inventory_array)
        elif self.obs_mode == 'frame_grid':
            obs = self.get_frame_grid_observation()
            # if all values in obs is 0, then print action and frame grid
            if all(value == 0 for value in obs.flatten()):
                print(f'Action : {action_tuple} \n Frame Grid : \n{self.curr_frame_grid}')
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
        Used in step to temporarily forward check given action whether structure is connected (support and target load)
        Input : frame center coordinates (frame_x, frame_y)
        Output : True if single target load is met
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

        # Check if all target loads are met
        if all(temp_target_loads_met.values()):
            return True
        else:
            return False
        
    def check_is_connected_multiple(self, frame_x, frame_y):
        '''
        Used in step to temporarily forward check given action whether structure is connected (support and target load)
        Input : frame center coordinates (frame_x, frame_y)
        Output : list of boolean values for each target load
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

        return temp_target_loads_met
    
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
        self.update_target_meet_multiple(new_frame) # updates self.target_loads_met and self.is_connected_fraction
    
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

    def update_frame_grid(self, new_frame, t_load_mag=None):
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
        
        # (Optional : add load magnitude to frame grid observation)
        # if new_frame.type_structure == FrameStructureType.EXTERNAL_FORCE: # target frame
        #     assert t_load_mag is not None, "Target load magnitude must be provided for target frame"
        #     # stack additional cells in pos y direction according to load magnitude ex) 100kN load -> 0 additional cells, 200KN -> 1 additional cell ...
        #     num_additional_cells = int(abs(t_load_mag[1]) / 100) - 1 # 100kN per cell
        #     assert new_frame.y_frame + num_additional_cells < 5, "Stacked external load exceeds frame. Decrease the load or place the load lower." # make sure that the additional cells are within the frame grid bounds
        #     for i in range(num_additional_cells+1):
        #         self.curr_frame_grid[new_frame.x_frame, new_frame.y_frame + i] = FrameStructureType.EXTERNAL_FORCE.idx

        # Add adjacent cells to valid position (if frame not load frame)
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

    def update_target_meet_multiple(self, new_frame):
        '''
        Used in apply_action
        Input : new_frame 
        Check if any of target load is met with addition of new frame(top right or top left node of frame meets with external load node) 
        Update self.target_loads_met and self.is_connected_fraction in place
        '''
        top_right = (new_frame.x + self.frame_size//2, new_frame.y + self.frame_size//2)
        top_left = (new_frame.x - self.frame_size//2, new_frame.y + self.frame_size//2)
        for target in self.target_loads_met:
            if top_right == target:
                self.target_loads_met[target] = True
            if top_left == target:
                self.target_loads_met[target] = True
        # update is_connected_fraction 
        # get list of boolean values from target_loads_met and calculate fraction of True values
        self.is_connected_fraction = sum(self.target_loads_met.values()) / len(self.target_loads_met)

    ## Drawing
    def draw_truss_analysis(self):
        '''
        Used within take step after episode as ended with connection
        Given that displacement has been updated
        Overlay displaced truss to plot by updating self.fig and self.ax based on self.curr_fea_graph.displacement
        Overlay failed elements in red based on self.curr_fea_graph.failed_elements
        '''
        displaced_truss_color = 'gray'
        disp_vis_scale = 2 # scale displacement for visualization 
        
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
        for edge, force in self.curr_fea_graph.failed_elements:
            start_id, end_id = edge
            start_coord = displaced_vertices[start_id]
            end_coord = displaced_vertices[end_id]
            if force >= 0.0: # tension
                self.ax.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]],
                    color='red', linestyle='-', linewidth=2)
            else:
                self.ax.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]],
                    color='blue', linestyle='-', linewidth=2)
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
        # Draw circle around max delfected node max_dixp = (V.id, d_mag) 
        max_v_id, max_d_mag = max_disp
        max_x_new, max_y_new = displaced_vertices[max_v_id]
        self.ax.add_patch(patches.Circle((max_x_new, max_y_new), radius=0.2, color='red', alpha=0.5))

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
                # Create a triangle with the top point at the vertex coordinate
                triangle_vertices = [
                    (coord[0], coord[1]),  # Top point
                    (coord[0] - 0.15, coord[1] - 0.2),  # Bottom-left point
                    (coord[0] + 0.15, coord[1] - 0.2)   # Bottom-right point
                ]
                self.ax.add_patch(patches.Polygon(triangle_vertices, color='black', lw=1.5, fill=True))
            else:
                self.ax.add_patch(patches.Circle(coord, radius=0.1, color='black', lw=0.5, fill=True ))
            self.ax.text(coord[0]-text_offset, coord[1]+text_offset, 
                         str(vertex.id), 
                         fontsize=10, ha='right', color='black')
            
        # Draw frames
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

        # random frame (red highlight)
        # for act in self.rand_init_actions:
        #     end_bool, frame_type, frame_x, frame_y = self.action_converter.decode(act)
        #     x , y = self.framegrid_to_board(frame_x, frame_y)
        #     rect = patches.Rectangle((x - self.frame_size//2, y - self.frame_size//2), 
        #                             self.frame_size, self.frame_size, 
        #                             linewidth=0, facecolor=(1, 0, 0, 0.2))
        #     self.ax.add_patch(rect)
            

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
        self.ax.clear() # TODO debug existing rect patches showing

        if self.render_mode == "human_playable":
            # Connect the button press event (add frame)
            self.click_event_id = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            # Connect the keypress event (select frame type)
            self.key_event_id = self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)

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
        if self.eps_terminate_valid == False:
            for frame_x, frame_y in self.valid_pos:
                x , y = self.framegrid_to_board(frame_x, frame_y)
                rect = patches.Rectangle((x - self.frame_size//2, y - self.frame_size//2), 
                                        self.frame_size, self.frame_size, 
                                        linewidth=0, edgecolor='lightblue', facecolor='lightblue')
                self.ax.add_patch(rect)
            
        # Draw current fea graph
        self.draw_fea_graph() # update self.fig, self.ax with current fea graph 

        # Overlay with displacement graph
        if self.eps_terminate_valid == True:
        # if len(self.curr_fea_graph.displacement) != 0 : # check if displacement has been analyzed 
            self.draw_truss_analysis() # last plot has displaced structure 
            # self.ax.text(
            #                 0.5, -0.05,  # x=0.5 centers the text, y=0.01 places it at the bottom
            #                 # f'Allowable Deflection : {self.allowable_deflection:.4f} m',
            #                 f'Allowable Deflection : {self.allowable_deflection:.3f} m, Max Deflection: {self.max_deflection:.3f} m, Reward: {self.episode_return:.1f}, Episode Length: {self.episode_length}',
            #                 color='black',
            #                 fontsize=14,
            #                 ha='left',  # aligns the text horizontally
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
                ha='left',  # aligns the text horizontally
                transform=self.ax.transAxes  # Use axis coordinates
            )

            self.ax.text(
                0.275, -0.05,  # Adjust x to position the value correctly
                f'{self.allowable_deflection:.3f} m',
                color='gray',
                fontsize=caption_fontsize_small,
                ha='left',  # aligns the text horizontally
                transform=self.ax.transAxes  # Use axis coordinates
            )

            self.ax.text(
                0.375, -0.05,  # Adjust x to position the text correctly
                'Max Deflection:',
                color='black',
                fontsize=caption_fontsize_small,
                ha='left',  # aligns the text horizontally
                transform=self.ax.transAxes  # Use axis coordinates
            )

            if self.max_deflection >= self.allowable_deflection:
                self.ax.text(
                    0.50, -0.05,  # Adjust x to position the value correctly
                    f'{self.max_deflection:.3f} m',
                    color='red',
                    fontsize=caption_fontsize_large,
                    ha='left',  # aligns the text horizontally
                    transform=self.ax.transAxes  # Use axis coordinates
                )
            else:
                self.ax.text(
                    0.525, -0.05,  # Adjust x to position the value correctly
                    f'{self.max_deflection:.3f} m',
                    color='green',
                    fontsize=caption_fontsize_large,
                    ha='left',  # aligns the text horizontally
                    transform=self.ax.transAxes  # Use axis coordinates
                )

            self.ax.text(
                0.625, -0.05,  # Adjust x to position the text correctly
                'Reward:',
                color='black',
                fontsize=caption_fontsize_small,
                ha='left',  # aligns the text horizontally
                transform=self.ax.transAxes  # Use axis coordinates
            )

            self.ax.text(
                0.7, -0.05,  # Adjust x to position the value correctly
                f'{self.episode_return:.1f}',
                color='green',
                fontsize=caption_fontsize_large,
                ha='left',  # aligns the text horizontally
                transform=self.ax.transAxes  # Use axis coordinates
            )

            self.ax.text(
                0.825, -0.05,  # Adjust x to position the text correctly
                'Episode Length:',
                color='black',
                fontsize=caption_fontsize_small,
                ha='left',  # aligns the text horizontally
                transform=self.ax.transAxes  # Use axis coordinates
            )

            self.ax.text(
                0.95, -0.05,  # Adjust x to position the value correctly
                f'{self.episode_length}',
                color='gray',
                fontsize=caption_fontsize_small,
                ha='left',  # aligns the text horizontally
                transform=self.ax.transAxes  # Use axis coordinates
            )
            # New line of text
            self.ax.text(
                0.1, -0.10,  # Adjust x and y to position the new text correctly
                'Inventory:',
                color='black',
                fontsize=caption_fontsize_small,
                ha='left',  # aligns the text horizontally
                transform=self.ax.transAxes  # Use axis coordinates
            )

            self.ax.text(
                0.20, -0.10,  # Adjust x and y to position the new value correctly
                f'light ({self.bc_inventory[FrameStructureType.LIGHT_FREE_FRAME]})     medium ({self.bc_inventory[FrameStructureType.MEDIUM_FREE_FRAME]})',
                color='gray',
                fontsize=caption_fontsize_small,
                ha='left',  # aligns the text horizontally
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
            if self.eps_terminate_valid:
                self.render_frame()
                render_path = os.path.join(self.render_dir, render_name)
                # Save the render
                self.render_frame()
                plt.savefig(render_path, bbox_inches='tight')
                plt.close(self.fig)
                # Increment the counter for the next file
                self.render_counter += 1
        elif self.render_mode == "rgb_end_interval":
            render_name = f"render{self.render_counter}_eps{self.global_terminated_episodes}_step{self.global_steps}.png" 
            if self.eps_terminate_valid and self.global_terminated_episodes % self.render_interval_eps < self.render_interval_consecutive:
                render_path = os.path.join(self.render_dir, render_name)
                # Save the render
                self.render_frame()
                plt.savefig(render_path, bbox_inches='tight')
                plt.close(self.fig)
                # Increment the counter for the next file
                self.render_counter += 1

        elif self.render_mode == "debug_valid":
            if self.render_valid_action:
                self.render_frame()
                plt.show()
                plt.close(self.fig)
        elif self.render_mode == "debug_end":
            if self.eps_terminate_valid:
                self.render_frame()
                plt.show()
                plt.close(self.fig)
        elif self.render_mode == "debug_all":
            self.render_frame() # initialize and update self.ax, self.fig object
            plt.show()
            plt.close(self.fig)

        elif self.render_mode == "human_playable":
            self.render_frame() # initialize and update self.ax, self.fig object
            # human_action values are changed 

            # save fig to render_list for video saving
            img = self.fig_to_rgb_array(self.fig)
            self.render_list.append(img)
            # show frame 
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
        if self.render_mode == "rgb_list" or self.render_mode == "human_playable":
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

    # Human Playable Mode
    def on_click(self, event):
        '''
        Function triggered at click in human playable mode
        Change current human action frame location based on snapped click location.
        cursor location is the centroid of the truss frame.
        Take next action to add frame to the grid.
        '''
        # Check if click is within the grid bounds 
        if event.xdata is not None and event.ydata is not None:
            # Get the frame grid coordinates of the cursor location
            frame_x = int(event.xdata // self.frame_size)
            frame_y = int(event.ydata // self.frame_size)
            # check if valid position 
            if (frame_x, frame_y) in self.valid_pos:
                self.human_action_frame_coords = (frame_x, frame_y) 
                print(f"human selected Frame grid coordinates: ({frame_x}, {frame_y})")
        # if event.inaxes != self.done_button_ax and event.xdata is not None and event.ydata is not None:
        

    
    def on_keypress(self, event):
        '''
        Function triggered at key press in human playable mode
        Change current human action frame type based on key press.
        '''
        if event.key == '1':
            self.human_action_frame_type = FrameStructureType.LIGHT_FREE_FRAME
        if event.key == '2':
            self.human_action_frame_type = FrameStructureType.MEDIUM_FREE_FRAME
        if event.key == 'e':
            self.human_action_end = True
        if event.key == 'c':
            self.human_action_end = False
        # TODO update frame type text in fig 
        self.fig.canvas.draw()

    def get_clicked_action(self):
        '''
        called in human_play mode to get action based on human click and key press
        return action integer based on human action frame coords and type
        '''
        action = self.action_converter.encode((self.human_action_end, self.human_action_frame_type.idx-1, self.human_action_frame_coords[0], self.human_action_frame_coords[1]))

        return action
    
    # Debugging
    def print_framegrid(self):
        '''
        Prints the current frame grid in a human-readable format with the x-axis across different lines.
        '''
        print("Current Frame Grid:")
        transposed_grid = self.curr_frame_grid.T  # Transpose the grid
        for row in reversed(transposed_grid):
            print(" ".join(map(str, row)))

    # Action Mask 
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
            if self.inventory_dict[freeframe_type] > 0:
                for frame_x, frame_y in self.valid_pos:
                    # forward look ahead to check if support and target loads are connected
                    temp_is_connected = self.check_is_connected(frame_x, frame_y)
                    if temp_is_connected == True:
                        for end_bool in [False, True]:
                            valid_actions.append((end_bool, freeframe_idx, frame_x, frame_y))

                    # forward look ahead to check if support and target loads are connected (TODO multiple loads)
                    # temp_connected_list = self.check_is_connected_multiple(frame_x, frame_y) # list of boolean values for each target
                    # if all(temp_connected_list) == True:
                    #     for end_bool in [False, True]:
                    #         valid_actions.append((end_bool, freeframe_idx, frame_x, frame_y))
                    
                    else:
                        end_bool = False
                        valid_actions.append((end_bool, freeframe_idx, frame_x, frame_y))
        
        if len(valid_actions) == 0:
            print(f'valid actions are empty!')
            print(f'inventory_dict : {self.inventory_dict}')
            print(f'valid_pos : {self.valid_pos}')
            self.print_framegrid()
            print(f'temp_is_connected : {temp_is_connected}')

        # encode action vectors to action integers using self.action_converter.encode(action)
        valid_action_ints = [self.action_converter.encode(action) for action in valid_actions]
        # print(f'valid action idx : {valid_action_ints}')
        
        # apply 1 to valid actions and 0 to invalid actions
        action_mask[valid_action_ints] = 1
        # action_mask[np.logical_not(action_mask)] = 0
        
        return action_mask
    
    # Random action Helper
    def add_rand_action(self, action_ind):
        '''
        Used in training to store random actions taken at initialization in self.rand_init_actions
        later used for random frame visualization in draw_fea_graph()
        '''
        if isinstance(action_ind, torch.Tensor):
            action = action_ind.cpu().numpy()
        self.rand_init_actions.append(action_ind)

    # Observation Helper
    def get_frame_grid_observation(self):
        '''
        Used in step and reset to return frame grid observation
        stack curr_frame_grid with current inventory
        '''
        # Get the number of columns in curr_frame_grid
        num_cols = self.curr_frame_grid.shape[0]

        # Create a new row with zeros
        # new_frame_row = np.zeros(num_cols, dtype=np.int64)
        light_frame_row = np.zeros(num_cols, dtype=np.int64)
        med_frame_row = np.zeros(num_cols, dtype=np.int64)
        
        # Get inventory value of medium frame 
        light_inventory = list(self.inventory_dict.values())[0]
        med_inventory = list(self.inventory_dict.values())[1]
        
        # Copy inventory_array values into the new rows
        # new_frame_row[:med_inventory] = 4 # fixed value for inventory of medium frame
        light_frame_row[:light_inventory] = 4  # fixed value for inventory of light frame
        med_frame_row[:med_inventory] = 5  # fixed value for inventory of medium frame

        # Expand dimensions to match the shape for stacking
        # new_frame_row = np.expand_dims(new_frame_row, axis=-1)
        light_frame_row = np.expand_dims(light_frame_row, axis=-1)
        med_frame_row = np.expand_dims(med_frame_row, axis=-1)


        # Append the new row to curr_frame_grid
        # obs = np.hstack([self.curr_frame_grid, new_frame_row]) # confusing bc obs array is frame transpose!
        obs = np.hstack([self.curr_frame_grid, light_frame_row, med_frame_row])  # confusing bc obs array is frame transpose!

        return obs