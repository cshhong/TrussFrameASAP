'''
Multiple target mode
'''
import gymnasium as gym
# from gym.spaces import Box, Discrete
from gymnasium.spaces import Box, Discrete, Graph
# from gymnasium.spaces.graph import *

import sys
import os

# Add the current directory to sys.path (to call TrussFrameMechanics )
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# sys.path.append("/Users/chong/Dropbox/2024Fall/TrussframeASAP")  # ensure Python can locate the TrussFrameASAP module 

# Get the absolute path of the current file (cantileverenv_v0.py)
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f'Parent Directory of cantilever env v2 added to sys.path : {PARENT_DIR}')

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
import matplotlib
matplotlib.use('Agg')  # prevent python memory accumulate


import numpy as np
import torch
import copy

import random

from collections import deque # for path finding
import itertools
import gc
import csv # baseline mode

class CantileverEnv_2(gym.Env):
    '''
        use - gymnasium.make("CartPole-v1", render_mode="human")

        Initialize the environment with a specified observation mode.
        Observation Modes: 
        - 'frame_grid': Only use the frame grid.
        - 'fea_graph': Only use the FEA graph.
        
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
            
            (obs_mode)'frame_grid' : frame_grid representation + added row with inventory values at end of row

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
                    edge_type_dict : dictionary where key : edge type int (weakest -> strongest), value : (outer diameter, inner wall thickness ratio) 
                    edges_dict : dictionary where keys : tuple of Vertex objects (v_1, v_2) and value : (outer diameter, inner wall thickness ratio)
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
            
    '''
    metadata = {"render_modes": [None, "debug_all", "debug_valid", "rgb_list", "debug_end", "rgb_end", "rgb_end_interval", "rgb_end_interval_nofail", "human_playable"], 
                "render_fps": 1,
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
                 env_idx = 0,
                 rand_init_seed = None,
                 bc_height_options=[1,2],
                 bc_length_options=[3,4,5],
                 bc_loadmag_options=[300,400,500],
                 bc_inventory_options=[(10,10), (10,5), (5,5), (8,3)],
                 num_target_loads = 2,
                 bc_fixed = None,
                 vis_utilization = False,
                 baseline_mode=False,
                 baseline_csv_path = None,
                 baseline_eps_count = 10,
                 baseline_n_expand = 3,
                 baseline_n_permute = None,
                 render_from_csv_mode = False,
                 render_from_csv_path = None,
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
        if self.render_mode == "rgb_end" or self.render_mode == "rgb_end_interval" or self.render_mode == "rgb_end_interval_nofail":
            self.render_counter = 0
        self.render_interval_eps = render_interval_eps
        self.render_interval_consecutive = render_interval_consecutive # number of episodes to render consecutively at interval

        # Current State (Frames)
        self.frames=[] # stores TrussFrameRL objects in sequence of creation
        
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

        # if actions are taken outside of valid_pos, large negative reward is given.  
        self.valid_pos = set() # set of frame_grid coordinates (frame_x, frame_y) in which new frame can be placed 
        self.render_valid_action = False
        # if episode ends without being connected, large negative reward is given.
        self.target_loads_met = {} # dictionary of (center board coordinate of target frame, bool) indicating if target load is met
        self.is_connected = False # whether the support is connected to (all) the target loads,
        self.is_connected_fraction = 0 # fraction of target loads connected to support

        self.eps_terminate_valid = False # after applying action with end_bool=1, signify is episode end is valid, used in visualization

        self.disp_reward_scale = 1e2 # scale displacement reward to large positive reward

        # Used for Logging
        self.max_deflection = None # float max deflection of the structure after completion at FEA
        self.max_deflection_node_idx = None
        
        # Set in reset() when boundary conditions are set 
        self.bc_inventory = None # dictionary of inventory in order of free frame types this does not change after reset
        self.inventory_dict = None # dictionary of inventory in order of free frame types
        # e.g.
        #     inventory = {
        #     FrameStructureType.FST_10_10 : inventory value, # -1 indicate no limits
        #     FrameStructureType.FST_20_20 : inventory value,
        # }s
        self.n_all_inventory = None # total number of inventory

        # self.obs_converter = None # ObservationBijectiveMapping object, used to encode and decode observations
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
        self.num_target_loads = num_target_loads
        self.bc_fixed = bc_fixed # fixed boundary conditions used to condition actor, critic network
        self.target_x_bounds = None # (left, right) set at initBoundaryCondition to prevent going beyond target x

        self.initBoundaryConditions() # set self.allowable_deflection, self.inventory_dict, self.frames, self.curr_frame_grid, self.target_loads_met, FrameStructureType.EXTERNAL_FORCE.node_load
        self.set_action_converter(self.inventory_dict) # set self.action_converter
        self.set_gym_spaces(self.action_converter) # set self.observation_space, self.action_space

        self.bc_condition = None # target (frame_x, frame_y) concatenated. used to condition actor, critic network

        # Used for Path finding
        self.support_board = []
        self.target_support_board = []

        # Element utilization
        self.edge_utilization = None # list of (center_x, center_y, utilization, dir) for each edge, at termination
        self.vis_utilization = vis_utilization # visualizes edges > P90 utilization
        self.utilization_ninety_percentile = 0 # determined at termination
        self.utilization_min = 0
        self.utilization_max = 0
        self.utilization_median = 0 # determined at termination
        self.utilization_std = 0 # determined at termination

        print("Initialized Cantilever Env!")

        # Baseline mode
        self.baseline_mode = baseline_mode
        self.baseline_csv_path = baseline_csv_path # path to csv file to save baseline results
        self.baseline_eps_count = baseline_eps_count
        self.baseline_n_expand = baseline_n_expand
        self.baseline_n_permute = baseline_n_permute

        # Render CSV mode
        self.render_from_csv_mode = render_from_csv_mode # boolean to indicate if render csv mode is used
        self.render_from_csv_path = render_from_csv_path # path to csv file to load designs from


    def reset(self, seed=None, **kwargs):
        '''
        Create boundary condition within environment with 
            generate_bc.set_cantilever_env_framegrid(self.frame_grid_size_x)
        that returns support_frames, targetload_frames within the frame grid
        self.frames, self.valid_pos, self.curr_frame_grid, self.curr_fea_graph is updated

        '''
        # print('Resetting Env!')

        self.render_list = []
        self.episode_return = 0
        self.episode_length = 0
        
        # Reset the current state
        self.frames = []
        self.curr_frame_grid = np.zeros((self.frame_grid_size_x, self.frame_grid_size_y), dtype=np.int64)
        self.curr_fea_graph = FEAGraph() #FEAGraph object
        self.valid_pos = set()

        self.target_loads_met = {}

        # Reset the Vertex ID counter
        Vertex._id_counter = 1

        self.max_cantilever_length_f = 0 # used to limit random init in training function 
        # Set boundary conditions ; support, target load, inventory
        self.initBoundaryConditions() # set self.cantiliver_length_f , self.allowable_deflection, self.inventory_dict, self.frames, self.curr_frame_grid, self.target_loads_met, FrameStructureType.EXTERNAL_FORCE.node_load

        self.eps_terminate_valid = False


        self.rand_init_actions = [] # reset random init actions
        self.reset_env_bool = True # set to True to initialize random actions in training function

        obs = self.get_frame_grid_observation(condition=False)
        info = {} # no info to return

        self.render_valid_action = True # temporarily turn on to trigger render
        self.render()
        self.render_valid_action = False
        # print(f"valid pos : {self.valid_pos}")
        
        gc.collect() # garbage collection to prevent memory leak

        # Baseline mode
        if self.baseline_mode:
            # generate one set of random designs and save in baseline csv
            print(f'Generating random designs with n_expand : {self.baseline_n_expand} and n_permute : {self.baseline_n_permute}')
            for i in range(self.baseline_eps_count):
                print(f'Generating baseline design {i+1}/{self.baseline_eps_count}')
                self.generate_random_designs(self.baseline_n_expand, self.baseline_n_permute)

        # Render from csv mode
        if self.render_from_csv_mode:
            try: # Open the CSV file
                with open(self.render_from_csv_path, mode='r') as csv_file:
                    print(f"RESET Rendering from CSV file: {self.render_from_csv_path}")
                    csv_reader = csv.DictReader(csv_file)  # Use DictReader to access columns by name
                    
                    # Iterate through each row in the CSV file
                    for row in csv_reader:
                        # debug print Episode and Number of Failed Elements value
                        # print(f"Episode: {row['Episode']}, Number of Failed Elements: {row['Number of Failed Elements']} Utilization Max : {row['Utilization Max']}")
                        # Dynamically find the "Frame Grid" value
                        if "Frame Grid" in row:
                            frame_grid_string = row["Frame Grid"]
                            # convert the string representation of the frame grid to a numpy array
                            # Remove all brackets
                            frame_grid_string = frame_grid_string.replace('[', '').replace(']', '')
                            # Split and parse
                            frame_grid_array = np.fromstring(frame_grid_string, sep=' ').reshape(-1, 6)
                        else:
                            print("Error: 'Frame Grid' column not found in the CSV file.")
                            break
                        # set return value from Episode Reward column
                        if "Episode Reward" in row:
                            self.episode_return = float(row["Episode Reward"])
                        # Render the fixed frame grid using the extracted value
                        self.render_fixed_framegrid(frame_grid_array)
            # Handle file not found error
            except FileNotFoundError:
                print(f"Error: CSV file not found at {self.render_from_csv_path}")
            except Exception as e:
                print(f"An error occurred while rendering from CSV: {e}")

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
            - self.curr_frame_grid, self.curr_fea_graph, (populate with support and target frames)
            - self.target_loads_met : dictionary of (center coordinate of frame, bool) indicating if target load is met
            - FrameStructureType.EXTERNAL_FORCE.node_load (set load value)

        Set frame grid, frame graph, fea graph accordingly 
        Set self.support_board, self.target_support_board for logging and target_loads_met for checking if target is met
        '''
        # Get boundary conditions
        # support_frames : dictionary (x_frame, y_frame)  cell location within frame grid of support frames
        # targetload_frames : dictionary of ((x_frame,y_frame) : [x_forcemag, y_forcemag, z_forcemag] (force is applied in the negative y direction).
        # cantilever_length : length of cantilever in number of frames
        support_frames, targetload_frames, inventory_dict, max_cantilever_length_f = \
            generate_bc.set_multiple_cantilever_env_framegrid(
                                                                self.frame_grid_size_x,
                                                                height_options = self.bc_height_options,
                                                                length_options = self.bc_length_options,
                                                                magnitude_options = self.bc_loadmag_options,
                                                                inventory_options = self.bc_inventory_options,
                                                                num_target_loads = self.num_target_loads,
                                                                fixed_hlm=self.bc_fixed,
                                                            )
        
        self.max_cantilever_length_f = max_cantilever_length_f
        self.allowable_deflection = self.frame_length_m * max_cantilever_length_f / 120 # length of cantilever(m) / 120
        # self.extr_load_mag = list(targetload_frames.values())[0] # magnitude of external load in kN (x,y,z)
        self.inventory_dict = inventory_dict
        self.bc_inventory = copy.deepcopy(inventory_dict) # does not change after reset
        self.n_all_inventory = sum(inventory_dict.values())

        # set FrameStructureType.EXTERNAL_FORCE magnitude values TODO where is this used? 
        FrameStructureType.EXTERNAL_FORCE.node_load = list(targetload_frames.values())[0]

        target_condition = [item for (x_frame, y_frame), forces in targetload_frames.items() for item in (x_frame, y_frame, forces[1])] # list of (x_frame, y_frame, y_forcemag) of target loads
        inventory_condition = list(inventory_dict.values()) # list of inventory values
        target_condition_short = [item for (x_frame, y_frame), forces in targetload_frames.items() for item in (x_frame, y_frame)]
        # self.bc_condition = target_condition + inventory_condition
        self.bc_condition = target_condition_short

        self.csv_bc = target_condition
        self.csv_inventory = inventory_condition

        # used for Path finding
        self.support_board = [] # board coordinates of support frames
        self.target_support_board = [] # list of board coordinates of target support frames

        # set target_x_bounds 
        self.target_x_bounds = (min([coord[0] for coord in targetload_frames.keys()]), max([coord[0] for coord in targetload_frames.keys()]))

        # Init supports and targets in curr_frame_grid according to bc
        for s_frame_coords in support_frames: # * update_frame_grid requires support to be updated first
            s_board_coords = self.frame_to_board(*s_frame_coords) # convert from frame grid coords to board coords
            new_s_frame = TrussFrameRL(s_board_coords, type_structure=FrameStructureType.SUPPORT_FRAME)
            self.support_board.append(s_board_coords)
            self.frames.append(new_s_frame)
            self.update_frame_grid(new_s_frame)
            self.update_fea_graph(new_s_frame)

        for t_frame in targetload_frames.items():
            t_frame_coord, t_load_mag = t_frame # (x,y) on frame grid, magnitude in kN
            # convert from frame grid coords to board coords 
            t_center_board = self.frame_to_board(*t_frame_coord) # center of proxy frame
            new_t_frame = TrussFrameRL(t_center_board, type_structure=FrameStructureType.EXTERNAL_FORCE)
            
            self.update_frame_grid(new_t_frame, t_load_mag=t_load_mag)
            self.update_fea_graph(new_t_frame, t_load_mag) 
            self.target_loads_met[t_center_board] = False
            
            # init light frame under target
            t_support_center_board = (t_center_board[0], t_center_board[1] - 2)
            new_t_frame_support = TrussFrameRL(t_support_center_board, type_structure=FrameStructureType.default_type)
            self.target_support_board.append(t_support_center_board)
            if t_support_center_board not in self.support_board:
                self.frames.append(new_t_frame_support)
                self.update_inventory_dict(new_t_frame_support)
                self.update_frame_grid(new_t_frame_support)
                self.update_fea_graph(new_t_frame_support)

    def set_action_converter(self, inventory_dict):
        '''
        set self.action_converter 
        Assume that self.inventory_dict is set
        '''
        assert isinstance(inventory_dict, dict), "inventory_dict must be a dictionary"
        # Set observation converter
        freeframe_inv_cap = [value for value in inventory_dict.values()] # get sequence of inventory level for freeframe types
        print(f'Freeframe inventory array: {freeframe_inv_cap}')  # Output: [1, 2, 3]

        # Set action converter
        freeframe_idx_min, freeframe_idx_max = FrameStructureType.get_freeframe_idx_bounds()
        self.action_converter = convert_gymspaces.ActionBijectiveMapping(self.frame_grid_size_x, 
                                                                            self.frame_grid_size_y,
                                                                        freeframe_idx_min, 
                                                                        freeframe_idx_max)
        
    def set_gym_spaces(self, action_converter=None):
        '''
        set self.observation_space, self.action_space according to obs mode
        Assumes that observation and action converters are provided as input
        observation space is continuous (Box) and action space is discrete (Discrete)

        Input
            action_converter : ActionBijectiveMapping object
        '''
        # # Set observation_space and action_space (following Gymnasium)
        # Define Observations : frame grid with extra row with medium inventory values
        # self.observation_space = Box(low=-1, high=4, shape=(self.frame_grid_size_x, self.frame_grid_size_y+len(FrameStructureType.get_free_frame_types())), dtype=np.int64)
        self.observation_space = Box(low=0, high=20, shape=(self.frame_grid_size_x, self.frame_grid_size_y), dtype=np.int64)
        self.single_observation_space = self.observation_space
        print(f'Obs Space : {self.observation_space} | Single obs space : {self.single_observation_space}')

        # Define Actions : end boolean, freeframe_type, frame_x, frame_y
        n_actions = action_converter._calculate_total_space_size()
        self.action_space = Discrete(n=n_actions, seed=self.rand_init_seed)
        print(f'Action Space : {self.action_space} sampling seed : {self.rand_init_seed}')
        self.single_action_space = self.action_space


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
        if self.baseline_mode == True:
            # already sampled in reset(), only need to terminate
            obs, reward, terminated, truncated, info = None, 0, True, False, {} #Debug is it ok if obs is None?
            self.eps_terminate_valid = True
            self.render()
            return obs, reward, terminated, truncated, info
        
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
        
        # interim target REWARD to encourage connecting to target loads
        # reward += self.is_connected_fraction * 0.025 # for each step once one/two target is connected
        reward += self.is_connected_fraction * 0.0025 # for each step once one/two target is connected
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

            # from self.bc_inventory and self.inventory_dict for each matching frame type get used frame count dictionary
            used_frame_count = {
                frame_type: self.bc_inventory[frame_type] - self.inventory_dict[frame_type]
                for frame_type in self.bc_inventory
            }

            # Inventory penalty capped 
            # from used_frame_count dictionary get sum of values and from self.bc_inventory get sum of values
            # reward -= 2*(sum(used_frame_count.values())/sum(self.bc_inventory.values())) # penalty for using more frames
            reward -= 3*(sum(used_frame_count.values())/sum(self.bc_inventory.values())) # penalty for using more frames
            # separate inventory penalty for frame type
            # get sum of ratio of used_frame_count / bc_inventory for each frame type

            # Max deflection penalty
            # if self.max_deflection < self.allowable_deflection:
            #     reward += np.log(self.allowable_deflection / self.max_deflection)  # large reward for low deflection e.g. 0.5 / 0.01 = 50, scale for allowable displacement considering varying bc 
            if self.max_deflection >= self.allowable_deflection:
                reward -= 1 # penalty for exceeding max deflection
                # reward -= 3 # penalty for exceeding max deflection

            reward -= len(self.curr_fea_graph.failed_elements) # large penalty by number of failed elements 
            # reward -= 2*len(self.curr_fea_graph.failed_elements) # large penalty by number of failed elements 

            # Explicit 90P Utilization reward - if the frame count is the same, design with higher utilization is superior
            if len(self.curr_fea_graph.failed_elements) == 0:
                reward += 2 * round(self.utilization_ninety_percentile/100, 2) # reward for utilization at 90P percentile
            
            # reward += self.num_target_loads * 2 # completion reward (long horizon)
            reward += self.num_target_loads * 3 # completion reward (long horizon)
            
        if truncated and not terminated:
            reward = 0 # Fixed value for episodes that truncate

        # Render frame
        self.render_valid_action = True

        # Store trajectory
        # inventory_array = np.array(list(self.inventory_dict.values())) # convert inventory dictionary to array in order of free frame types
        
        obs = self.get_frame_grid_observation(condition=False)
        # print(f'{obs=}')
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
        # print(f'reward : {reward} \n Terminated : {terminated} \n Truncated : {truncated} \n Info : {info}')
        return obs, reward, terminated, truncated, info
        
    
    def check_is_connected_multiple(self, frame_x, frame_y):
        '''
        Used in get_action_mask to forward check given action whether structure is connected (support and target load)
        unidirectional, so only have to check if current frame is below target
        Input : frame center coordinates (frame_x, frame_y)
        Output : copy of self.target_loads_met with updated boolean values
        '''
        # # check if top-right, or top-left node of frame changes current self.target_loads_met values
        # # given temporary changed values, if all are true, return True
        temp_target_loads_met = self.target_loads_met.copy()
        center = self.frame_to_board(frame_x, frame_y) # get center board coordinates
        for target in self.target_loads_met:
            # if center is below target
            if (center[1] == target[1] - 2) and (center[0] == target[0]):
                temp_target_loads_met[target] = True
        return temp_target_loads_met
    
    def check_is_connected_bidirectional_temp(self, frame_x, frame_y):
        '''
        Used in get_action_mask to temporarily forward check given action whether structure is connected (support and target load)
        bidirectional, so have to path find with current frames in board 
        Input : frame center coordinates (frame_x, frame_y)
        Output : copy of self.target_loads_met with updated boolean values
        '''
        temp_target_loads_met = copy.deepcopy(self.target_loads_met)
        # check targets that are not connected 
        unconnected_targets = [target for target, met in temp_target_loads_met.items() if not met]
        # start from support, find path to target_support given current frames
        support_board = [coord for tup in self.support_board for coord in tup] # unpack support_board list and tuple
        support_frame = self.board_to_frame(*support_board)

        for target in unconnected_targets:
            target_support_frame = self.board_to_frame(target[0], target[1]-2)
            temp_connected = self.check_connected_path_temp(support_frame, target_support_frame, frame_x, frame_y)
            if temp_connected:
                temp_target_loads_met[target] = True
                 # function to check if current board has path from support to target load
        return temp_target_loads_met
    
    def check_connected_path_temp(self, support_frame, target_support_frame, frame_x=None, frame_y=None):
        '''
        Lookahead to check if there is path from support to target load, given hypothetical frame
        '''
        free_frame_types = FrameStructureType.get_free_frame_types() # get free frame types
        free_frame_types_idx = [frame_type.idx for frame_type in free_frame_types] # get free frame types indices

        # create temporary frame grid with new frame
        temp_frame_grid = copy.deepcopy(self.curr_frame_grid) # Deep copy of the current frame grid
        if frame_x is not None and frame_y is not None:
            temp_frame_grid[frame_x][frame_y] = free_frame_types_idx[0]  # Add the hypothetical frame (first of free frame types)

        # BFS setup
        queue = deque()
        visited = set()

        start_x, start_y = support_frame
        goal_x, goal_y = target_support_frame

        queue.append((start_x, start_y))
        visited.add((start_x, start_y))

        # 4-directional movement (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            x, y = queue.popleft()
            if (x, y) == (goal_x, goal_y):
                return True

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.frame_grid_size_x) and (0 <= ny < self.frame_grid_size_y):
                    if (nx, ny) not in visited:
                        if temp_frame_grid[nx][ny] in free_frame_types_idx or (nx, ny) == (goal_x, goal_y):
                            queue.append((nx, ny))
                            visited.add((nx, ny))

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
        frame_center = self.frame_to_board(frame_x, frame_y)
        # print(f"Applying Action with  freeframe_idx : {freeframe_idx} at {frame_center}")
        frame_structure_type = FrameStructureType.get_framestructuretype_from_idx(freeframe_idx)
        new_frame = TrussFrameRL(pos = frame_center, type_structure=frame_structure_type)
        
        # update current state 
        self.update_inventory_dict(new_frame)
        self.update_frame_grid(new_frame)
        self.update_fea_graph(new_frame)

        if end == 1: # update displacement info in fea graph if episode end
            self.update_displacement() # updates self.max_deflection, self.max_deflection_node_idx
            self.update_utilization() # updates self.max_utilization
        
        self.frames.append(new_frame)

        self.update_target_loads_met_bidirectional() # updates self.target_loads_met and self.is_connected_fraction
    
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

    def update_frame_grid(self, new_frame, t_load_mag=None):
        '''
        Given new frame object, update current frame grid where 
        Input
            new_frame : TrussFrameRL object 
            t_load_mag : target load magnitude in kN (x,y,z) if new_frame is target load frame
        Updates:
        - self.curr_frame_grid : A grid where each cell is updated based on the frame type.
            - (cell state for array size frame_grid_size_x frame_grid_size_y) 
            - grid where each cell has value force = -1, unoccupied= 0, support frame = 1, free frame (light) = 2, free frame (med) = 3, inventory (light) = 4, inventory (med) = 5
            - * light frame is added below target load frame
        - self.valid_pos : A set of valid (x, y) frame positions on the frame grid where a new frame can be placed.
        '''
        # Update the current frame grid with the new frame's type
        if new_frame.type_structure.idx == FrameStructureType.SUPPORT_FRAME.idx:
            self.curr_frame_grid[new_frame.x_frame, new_frame.y_frame] = new_frame.type_structure.idx
        else:
            if self.curr_frame_grid[new_frame.x_frame, new_frame.y_frame] != FrameStructureType.SUPPORT_FRAME.idx:
                self.curr_frame_grid[new_frame.x_frame, new_frame.y_frame] = new_frame.type_structure.idx
            else:
                raise ValueError(f"Trying to add a frame at a support position")

        # Remove the position of the new frame from valid_pos if it exists
        if new_frame.type_structure.idx != FrameStructureType.SUPPORT_FRAME.idx and new_frame.type_structure.idx != FrameStructureType.EXTERNAL_FORCE.idx:
            if (new_frame.x_frame, new_frame.y_frame) in self.valid_pos:
                self.valid_pos.remove((new_frame.x_frame, new_frame.y_frame))
        # else:
        #     raise ValueError(f"Position ({new_frame.x_frame}, {new_frame.y_frame}) is not a valid position for placing a frame.")
        
        # (optional) Add load magnitude to frame grid observation
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
                    # check if adjacent cell is in target x bounds
                    if self.target_x_bounds[0] <= x_adj and x_adj <= self.target_x_bounds[1]:
                        # check if adjacent cell is not occupied
                        if self.curr_frame_grid[x_adj, y_adj] == FrameStructureType.UNOCCUPIED.idx or \
                            self.curr_frame_grid[x_adj, y_adj] == FrameStructureType.EXTERNAL_FORCE.idx:
                            self.valid_pos.add((x_adj, y_adj))

    # def _update_fea_graph(self, new_frame, t_load_mag=[0.0, 0.0, 0.0]):
    #     '''
    #     Input 
    #         new_frame : TrussFrameRL object (centroid, frame type)

    #     Given new TrussFrameRL object, update self.curr_feagraph (FEAgraph object) where 
    #         self.vertices : dictionary of vertices with coordinate as key and Vertex object as value
    #         self.edges : adjacency list of tuples of vertex indices pairs
    #         self.maximal_edges : dictionary where key is direction and value is list of MaximalEdge objects 
    #         {
    #             'horizontal': [],
    #             'vertical': [],
    #             'LB_RT': [],
    #             'LT_RB': []
    #         }
    #         self.load : A list of tuples (node.id, [load.x, load.y, load.z]) 

    #     Update current FEAgraph with added truss frame so that existing node indices are preserved
    #     1. merge overlapping new nodes with existing nodes
    #     2. check line overlap with existing edge using maximal edge representation 
    #     3. update edge list with new line segments

    #     '''
    #     # Calculate the positions of the four vertices
    #     half_size = self.frame_size // 2
    #     vert_pos = [
    #         (new_frame.x - half_size, new_frame.y - half_size),  # Bottom-left
    #         (new_frame.x + half_size, new_frame.y - half_size),  # Bottom-right
    #         (new_frame.x + half_size, new_frame.y + half_size),  # Top-right
    #         (new_frame.x - half_size, new_frame.y + half_size)   # Top-left
    #     ]

    #     if new_frame.type_structure == FrameStructureType.EXTERNAL_FORCE: # target frame
    #         target_load_pos = vert_pos[0], vert_pos[1] # bottom right, top right vertices of target frame
    #         for pos in target_load_pos:
    #             self.curr_fea_graph.external_loads[pos] = [l / 2 for l in t_load_mag] # distribute load to two nodes
    #         # print(f'added external load to fea graph : {target_load_pos}')
    #     else: # SUPPORT_FRAME, FST_10_10, FST_20_20
    #         new_vertices = [] # Vertex object in order of bottom-left, bottom-right, top-right, top-left
    #         for i, pos in enumerate(vert_pos):
    #             # If new node overlaps with existing node, merge (preserve existing node attributes - id, is_free)
    #             if pos in self.curr_fea_graph.vertices:
    #                 new_v = self.curr_fea_graph.vertices[pos] # get overlapping existing node
    #                 # allow change free->fixed but not fixed->free
    #                 if new_frame.type_structure == FrameStructureType.SUPPORT_FRAME:
    #                     new_v.is_free = False
    #             else: # If node does not overlap with existing node, create new node 
    #                 is_free = None
    #                 if new_frame.type_structure == FrameStructureType.FST_10_10 or new_frame.type_structure == FrameStructureType.FST_20_20: # Free 
    #                     is_free = True
    #                 elif new_frame.type_structure == FrameStructureType.SUPPORT_FRAME: # Support
    #                     if i==0 or i==1: # Bottom left, Bottom right are fixed 
    #                         is_free = False
    #                         self.curr_fea_graph.supports.append(pos) # add to list of supports
    #                     else: # Top left, Top right are free
    #                         is_free = True
    #                 new_v = Vertex(pos, is_free=is_free, load=new_frame.type_structure.node_load)
                    
    #                 # additionally check if meets with external load, and if so, combine load
    #                 if pos in self.curr_fea_graph.external_loads:
    #                     new_v.load = [x + y for x, y in zip(new_v.load, self.curr_fea_graph.external_loads[pos])]

    #                 # add new node to fea graph
    #                 self.curr_fea_graph.vertices[pos] = new_v 
    #             # add to new vertices to combine edges                    
    #             new_vertices.append(new_v) 

    #         # Check line overlap with existing edge  
    #         self.curr_fea_graph.combine_and_merge_edges(frame_type_shape=new_frame.type_shape,new_vertices=new_vertices, frame_structure_type=new_frame.type_structure)

    def update_fea_graph(self, new_frame, t_load_mag=[0.0, 0.0, 0.0]):
        '''
        DEBUG to use FrameStructureType.is_free_nodes that boolean values for (bottom_left, bottom_right, top_left, top_right) nodes in frame
        Input 
            new_frame : TrussFrameRL object (centroid, frame type)

        Given new TrussFrameRL object, update self.curr_feagraph (FEAgraph object) where 
            self.vertices : dictionary of vertices with coordinate as key and Vertex object as value
            self.edges : adjacency list of tuples of vertex indices pairs
            self.load : A list of tuples (node.id, [load.x, load.y, load.z]) 

        Update current FEAgraph with added truss frame so that existing node indices are preserved
        1. merge overlapping new nodes with existing nodes
        2. check line overlap with existing edge using maximal edge representation 
        3. update edge list with new line segments

        '''
        # Non physical Frames
        # update external load position and magnitude separately
        if new_frame.type_structure == FrameStructureType.EXTERNAL_FORCE: # target frame
            # target_load_pos = vert_pos[0], vert_pos[1] # bottom right, top right vertices of target frame
            target_load_pos = new_frame.bottom_left, new_frame.bottom_right 
            for pos in target_load_pos:
                half_load = [l / 2 for l in t_load_mag]
                # print(f'added external load to fea graph : {pos} with load {half_load}')
                self.curr_fea_graph.external_loads[pos] = half_load # distribute load to two nodes
                # check existing nodes before adding new Vertex
                if pos in self.curr_fea_graph.vertices: # if node already exists, merge load
                    existing_v = self.curr_fea_graph.vertices[pos]
                    existing_v.load = [x + y for x, y in zip(existing_v.load, half_load)] # merge load
                else:
                    new_v = Vertex(pos, is_free=True, load=half_load)
                    self.curr_fea_graph.vertices[pos] = new_v # add to vertices
        
        # Physical Frames
        else:
            # update support positions separately
            if new_frame.type_structure == FrameStructureType.SUPPORT_FRAME: # support frame
                self.curr_fea_graph.supports.append(new_frame.bottom_left) # add to list of supports
                self.curr_fea_graph.supports.append(new_frame.bottom_right) # add to list of supports

            # create Vertex objects for 4 vertices in frame with check with overlap with existing 
            # get vertex position, is_free, load from frame object
            all_vert_pos = new_frame.get_vertex_positions() # get vertex positions of frame
            # print(f'all_vert_pos : {all_vert_pos}')
            all_vert_is_free = new_frame.type_structure.is_free_nodes # get is_free boolean values of frame
            # print(f'all_vert_is_free : {all_vert_is_free}')
            # list with new_frame.type_structure.node_load (x,y,z) tuple 4 times for each node
            all_vert_load = [new_frame.type_structure.node_load for _ in range(4)]
            # print(f'all_vert_load : {all_vert_load}')

            new_vertices = [] # Vertex object in order of bottom-left, bottom-right, top-right, top-left used to update edges
            for i in range(4):
                # handle existing node overlap case ; 
                    # fixed > free
                    # load sum 
                # If new node overlaps with existing node, merge (preserve existing node attributes - id, is_free)
                if tuple(all_vert_pos[i]) in self.curr_fea_graph.vertices:
                    # print(f'    checking overlapping node at {self.curr_fea_graph.vertices[all_vert_pos[i]]}')
                    # print(f'get old v : { self.curr_fea_graph.vertices[all_vert_pos[i]]}')
                    # print(f'    all vertices : {[v.id for v in self.curr_fea_graph.vertices.values()]}')
                    old_v = self.curr_fea_graph.vertices[all_vert_pos[i]] # get overlapping existing Vertex object and update
                    # print(f'    overlapping node {old_v}')
                    # allow change free->fixed but not fixed->free (if at least one of the existing nodes is fixed, then set new node to fixed)
                    if old_v.is_free == False or all_vert_is_free[i] == False: # if either node is fixed, set to fixed
                        old_v.is_free = False
                    # compare l2 norm of load vectors and use the larger one
                    old_v.load = old_v.load if sum(a**2 for a in old_v.load) >= sum(b**2 for b in all_vert_load[i]) else all_vert_load[i]
                    new_v = old_v # use existing node
                else:
                    new_v = Vertex(all_vert_pos[i], is_free=all_vert_is_free[i], load=all_vert_load[i]) 
                    # add new node to fea graph
                    self.curr_fea_graph.vertices[all_vert_pos[i]] = new_v 
                    # print(f'    creating new vertice {new_v}')

                new_vertices.append(new_v) # add to new vertices to combine edges
                
            # print(f'    all new vertices : {[v.id for v in new_vertices]}')


            # Check line overlap with existing edge  
            self.curr_fea_graph.combine_and_merge_edges(frame_type_shape=new_frame.type_shape,new_vertices=new_vertices, frame_structure_type=new_frame.type_structure)

            # print(f'updating fea graph with new frame {new_frame}')
            # print(f'{self.curr_fea_graph.vertices=}')

    def update_utilization(self):
        '''
        Called in apply_action at termination 
        Updates self.utilization_median, self.utilization_std, self.utilization_ninety_percentile
        '''
        # Get utlization value from edge_utilization
        self.edge_utilization = self.curr_fea_graph.get_element_utilization() # list of (center_x, center_y, utilization, dir) for each edge 
        util_val = [abs(edge[2])*100 for edge in self.edge_utilization]
        self.utilization_min = np.min(util_val)
        self.utilization_max = np.max(util_val)
        self.utilization_median = np.median(util_val)
        self.utilization_std = np.std(util_val)
        self.utilization_ninety_percentile = np.percentile(util_val, 90) # TODO should this exclude failed elements?

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
        curr_env = jl.seval('Base.active_project()')
        # print(f"The current active Julia environment is located at: {curr_env}")

        # Step 0: Initialize Julia session and import necessary Julia modules
        # jl.seval('using AsapToolkit')
        jl.seval('using Asap')

        # Include the Julia file using the absolute path
        truss_analysis_path = os.path.join(PARENT_DIR, "..", "TrussFrameMechanics", "truss_analysis.jl")
        jl.include(truss_analysis_path)
        jl.seval('using .TrussAnalysis')

        displacement, failed_elements, utilization = pythonAsap.solve_fea(jl, self.curr_fea_graph, self.frame_length_m) # return nodal displacement
        self.curr_fea_graph.displacement = displacement
        self.curr_fea_graph.failed_elements = failed_elements
        self.curr_fea_graph.utilization = utilization

        self.max_deflection_node_idx, self.max_deflection = self.curr_fea_graph.get_max_deflection() # update max_deflection


    def update_target_loads_met_bidirectional(self):
        '''
        given updated grid, update target loads that were not previously connected
        Update self.target_loads_met and self.is_connected_fraction in place
        '''
        unconnected_targets = [target for target, met in self.target_loads_met.items() if not met]
        
        # start from support, find path to target_support given current frames
        support_board = [coord for tup in self.support_board for coord in tup]# unpack support_board list and tuple
        support_frame = self.board_to_frame(*support_board)

        for target in unconnected_targets:
            target_support = self.board_to_frame(target[0], target[1]-2)
            temp_connected = self.check_connected_path_temp(support_frame, target_support)
            if temp_connected:
                self.target_loads_met[target] = True

        self.is_connected_fraction = sum(self.target_loads_met.values()) / len(self.target_loads_met)

    ## Drawing
    def draw_truss_analysis(self):
        '''
        Used within take step after episode as ended with connection
        Given that displacement has been updated
        Overlay displaced truss to plot by updating self.fig and self.ax based on self.curr_fea_graph.displacement
        Overlay failed elements in red based on self.curr_fea_graph.failed_elements

        uses 
            - self.curr_fea_graph.vertices 
            - self.curr_fea_graph.displacement
            - self.curr_fea_graph.edges 
            - self.curr_fea_graph.get_element_utilization()
            - self.curr_fea_graph.displacement
        '''
        displaced_truss_color = 'gray'
        disp_vis_scale = 1 # scale displacement for visualization 
        disp_vis_scale = 5 # scale displacement for visualization 
        
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
            # Add text showing displacement magnitude next to each circle if over allowable deflection
            if d_mag >= self.allowable_deflection:
                self.ax.text(new_x + 0.1, new_y + 0.1, f'{d_mag:.3f}', color='gray', fontsize=8)
                self.ax.add_patch(patches.Circle((new_x, new_y), radius=0.1, color='red', alpha=0.3))
        
        # Connect deflected nodes with edges
        if self.curr_fea_graph.edges == []: # uses edges_dict where keys : (v_1, v_2) Vertex objects, values : (outer diameter, inner wall thickness ratio) instead of edges list of (v_id_1, v_id_2) to get strong edges
            edges = [(e[0].id, e[1].id) for e in self.curr_fea_graph.edges_dict.keys()]
        else:
            edges = self.curr_fea_graph.edges
        for edge in edges:
            start_id, end_id = edge  # node ids
            try:
                start_coord = displaced_vertices[start_id]
            except KeyError:
                print(f"KeyError: {start_id} not in displaced_vertices \n {displaced_vertices}")
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
                    color='blue', linestyle='-', linewidth=2)
            else:
                self.ax.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]],
                    color='red', linestyle='-', linewidth=2)
                
        # # Highlight max displacement
        # # Add text to highlight the max displacement
        # maxd_x_coord, maxd_y_coord = displaced_vertices[max_disp[0]]
        # maxd_value = max_disp[1]
        # self.max_deflection = max_disp[1]
        
        # # Draw circle around max deflected node max_disp = (V.id, d_mag) 
        # max_v_id, max_d_mag = max_disp
        # max_x_new, max_y_new = displaced_vertices[max_v_id]
        # self.ax.add_patch(patches.Circle((max_x_new, max_y_new), radius=0.2, color='red', alpha=0.3))
        
        # Overlay utilization on each edge
        if self.vis_utilization == True:
            for edge in self.edge_utilization:
                center_x, center_y, util, dir = edge
                if util >= self.utilization_ninety_percentile:
                        # Determine placement of text depending on edge direction
                    if dir == 'H':  # Horizontal
                        text_x = center_x
                        text_y = center_y  # Place above the edge
                    elif dir == 'V':  # Vertical
                        text_x = center_x # Place to the right of the edge
                        text_y = center_y
                    elif dir == 'D_LB_RT':  # Diagonal left-bottom to right-top
                        text_x = center_x - 0.4
                        text_y = center_y - 0.4 # Place above and to the right
                    elif dir == 'D_LT_RB':  # Diagonal left-top to right-bottom
                        text_x = center_x + 0.4
                        text_y = center_y - 0.4  # Place below and to the right

                    self.ax.text(text_x, text_y, f'{abs(util)*100:.1f}', color='green', rotation=25, fontsize=9, ha='center', va='center')
        
    def draw_fea_graph(self):
        '''
        Update self.fig and self.ax based on self.curr_fea_graph
        used in render_frame
        uses 
            - self.curr_fea_graph.vertices to draw nodes
            - self.curr_fea_graph.edges_dict to add strong edges
            - self.curr_fea_graph.external_loads to draw external loads

        '''
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
                    (coord[0] - 0.25, coord[1] - 0.3),  # Bottom-left point
                    (coord[0] + 0.25, coord[1] - 0.3)   # Bottom-right point
                ]
                self.ax.add_patch(patches.Polygon(triangle_vertices, color='black', lw=1.5, fill=True))
            else:
                self.ax.add_patch(patches.Circle(coord, radius=0.1, color='black', lw=0.5, fill=True ))
            # vertex ind
            # self.ax.text(coord[0]-text_offset, coord[1]+text_offset, 
            #              str(vertex.id), 
            #              fontsize=10, ha='right', color='black')
            
        # Draw edges (alternative to frame)
        for v_pair, e_section in self.curr_fea_graph.edges_dict.items():
            start_v, end_v = v_pair
            outer_d, inward_thickness_ratio = e_section
            start_coord = start_v.coordinates
            end_coord = end_v.coordinates
            # Draw the line connecting the start and end vertices
            # linewidth= 10 * outer_d + 50 * (inward_thickness_ratio**2)
            linewidth= 10 * outer_d + 50 * (inward_thickness_ratio**2)
            self.ax.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]], color='black', linestyle='-', linewidth=linewidth)


        # random frame (red highlight)
        # for act in self.rand_init_actions:
        #     end_bool, frame_type, frame_x, frame_y = self.action_converter.decode(act)
        #     x , y = self.frame_to_board(frame_x, frame_y)
        #     rect = patches.Rectangle((x - self.frame_size//2, y - self.frame_size//2), 
        #                             self.frame_size, self.frame_size, 
        #                             linewidth=0, facecolor=(1, 0, 0, 0.2))
        #     self.ax.add_patch(rect)
            

        # Draw external forces as red arrows
        # self.curr_fea_graph.external_loads.items() are list of (coord, loads)
        # loads are distributed along bottom left and right nodes of target load frame
        # should give list of (mid coords, sum of loads) for all pairs in order
        # visualize load at halfway between two nodes
        # loads_vis = [
        #             (tuple((c1 + c2) / 2 for c1, c2 in zip(coord1, coord2)), tuple(l1 + l2 for l1, l2 in zip(load1, load2)))
        #             for (coord1, load1), (coord2, load2) in zip(
        #                 list(self.curr_fea_graph.external_loads.items())[::2], # even index
        #                 list(self.curr_fea_graph.external_loads.items())[1::2] # odd index
        #             )
        #             ]

        # Load arrows on node
        loads_vis = self.curr_fea_graph.external_loads.items()

        # for coord, loads in self.curr_fea_graph.external_loads.items():
        for coord, load in loads_vis:
            force_magnitude = (load[0]**2 + load[1]**2 + load[2]**2)**0.5
            if force_magnitude >= 0:
                if force_magnitude !=0:
                    arrow_dx = (load[0]) * 0.01 
                    # arrow_dy = (load[1]) * 0.01
                    arrow_dy = - 1.25
                    linestyle = '-'
                    linewidth = 2.0

                    arrow_tail_x = coord[0] - arrow_dx
                    arrow_tail_y = coord[1] - arrow_dy
                    # arrow_head_x = arrow_tail_x - arrow_dx
                    # arrow_head_y = arrow_tail_y - arrow_dy

                    if force_magnitude > 0:
                        self.ax.text(arrow_tail_x, arrow_tail_y + 0.1, f"{force_magnitude:.0f} kN", color='black', fontsize=12)

                    self.ax.arrow(arrow_tail_x, arrow_tail_y, arrow_dx, arrow_dy+0.2, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=linewidth, linestyle = linestyle)
    
                else: # no force but indicate target location (don't draw arrow)
                    arrow_dx = 0
                    arrow_dy = -1.0
                    linestyle =':'
                    linewidth = 1.0
    # Render 
    def render_frame(self):
        '''
        initialize and updates self.ax, self.fig object 
        uses
            - self.valid_pos
            - self.curr_fea_graph
            (draw_fea_graph)
            - self.curr_fea_graph.vertices to draw nodes
            - self.curr_fea_graph.edges_dict to draw edges
            - self.curr_fea_graph.external_loads to draw external loads
            (draw_truss_analysis)
            - self.curr_fea_graph.vertices 
            - self.curr_fea_graph.displacement
            - self.curr_fea_graph.edges_dict 
            - self.curr_fea_graph.get_element_utilization()
            - self.curr_fea_graph.displacement
        '''
        # Create the figure and axes
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.clear() # TODO debug existing rect patches showing

        if self.render_mode == "human_playable":
            # Connect the button press event (add frame)
            self.click_event_id = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            # Connect the keypress event (select frame type)
            self.key_event_id = self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)


        # Draw grid lines
        # for i in range(0, self.board_size_x + 1, 2):
        #     self.ax.axvline(x=i, color='lightblue', linestyle='-', linewidth=2, zorder=0)
        # for j in range(0, self.board_size_y + 1, 2):
        #     self.ax.axhline(y=j, color='lightblue', linestyle='-', linewidth=2, zorder=0)

        # Highlight valid position cells (except for final frame if terminated)
        if self.eps_terminate_valid == False:
            for frame_x, frame_y in self.valid_pos:
                x , y = self.frame_to_board(frame_x, frame_y)
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

            # Caption at termination
            caption_fontsize_large = 16
            caption_fontsize_small = 12
            # 0.1 0.25 0.325 0.45 0.55 0.65 0.75 0.85
            # Max Deflection text
            self.ax.text(
                0.0, -0.075,   
                'Max Deflection :',
                color='black',
                fontsize=caption_fontsize_small,
                ha='left',   
                transform=self.ax.transAxes  # Use axis coordinates
            )
            # Max Deflection value
            if self.max_deflection >= self.allowable_deflection:
                self.ax.text(
                    0.15, -0.075,     
                    f'{self.max_deflection:.3f} m',
                    color='red',
                    fontsize=caption_fontsize_small,
                    weight='bold',
                    ha='left',   
                    transform=self.ax.transAxes  # Use axis coordinates
                )
            else:
                self.ax.text(
                    0.15, -0.075,     
                    f'{self.max_deflection:.3f} m',
                    color='black',
                    fontsize=caption_fontsize_small,
                    weight='bold',
                    ha='left',   
                    transform=self.ax.transAxes  # Use axis coordinates
                )
            # Reward text
            self.ax.text(
                0.3, -0.075,   
                'Reward :',
                color='black',
                fontsize=caption_fontsize_small,
                ha='left',   
                transform=self.ax.transAxes  # Use axis coordinates
            )
            # Reward value
            self.ax.text(
                0.4, -0.075,  
                f'{self.episode_return:.3f}',
                color='black',
                weight='bold',
                fontsize=caption_fontsize_small,
                ha='left',   
                transform=self.ax.transAxes  # Use axis coordinates
            )

            # Failed Elements
            self.ax.text(
                0.475, -0.075,
                # f'Utilization Median (%), Std, P90 :',
                f'Failed elements :',
                color='black',
                fontsize=caption_fontsize_small,
                ha='left',
                transform=self.ax.transAxes  # Use axis coordinates
            )

            self.ax.text(
                0.60, -0.075,
                # f'{self.utilization_median:.1f} %, {self.utilization_std:.1f},  {self.utilization_ninety_percentile:.1f} ',
                f'{len(self.curr_fea_graph.failed_elements)} ',
                color='black',
                weight='bold',
                fontsize=caption_fontsize_small,
                ha='left',
                transform=self.ax.transAxes  # Use axis coordinates
            )
            # Utilization 
            self.ax.text(
                0.65, -0.075,
                # f'Utilization Median (%), Std, P90 :',
                f'Utilization P90:',
                color='black',
                fontsize=caption_fontsize_small,
                ha='left',
                transform=self.ax.transAxes  # Use axis coordinates
            )

            self.ax.text(
                0.80, -0.075,
                # f'{self.utilization_median:.1f} %, {self.utilization_std:.1f},  {self.utilization_ninety_percentile:.1f} ',
                f'{self.utilization_ninety_percentile:.1f} ',
                color='black',
                weight='bold',
                fontsize=caption_fontsize_small,
                ha='left',
                transform=self.ax.transAxes  # Use axis coordinates
            )

            # New line of text
            # Allowable Deflection text
            self.ax.text(
                0.0, -0.125,   
                'Allowable Deflection :',
                color='black',
                fontsize=caption_fontsize_small,
                ha='left',   
                transform=self.ax.transAxes  # Use axis coordinates
            )
            # Allowable Deflection value
            self.ax.text(
                0.175, -0.125,     
                f'{self.allowable_deflection:.3f} m',
                color='gray',
                fontsize=caption_fontsize_small,
                ha='left',   
                transform=self.ax.transAxes  # Use axis coordinates
            )
            # Frame count text
            self.ax.text(
                0.0, -0.175,  
                'Frame Count :',
                color='black',
                fontsize=caption_fontsize_small,
                ha='left',   
                transform=self.ax.transAxes  # Use axis coordinates
            )
            
            used_frame_count = {
                frame_type: self.bc_inventory[frame_type] - self.inventory_dict[frame_type]
                for frame_type in self.bc_inventory
            }

            total_frame_count = sum(used_frame_count.values())
            # create string with FrameStructureType names and their corresponding used frame count / total frame count
            frame_count_str = ',   '.join(
                f'{frame_type.name} ({used_frame_count[frame_type]} / {self.bc_inventory[frame_type]})'
                for frame_type in FrameStructureType.get_free_frame_types()
            )

            # Frame count text
            self.ax.text(
                0.125, -0.175, 
                f'{frame_count_str}     Total ({total_frame_count} / {sum(self.bc_inventory.values())})',
                color='gray',
                fontsize=caption_fontsize_small,
                ha='left',   
                transform=self.ax.transAxes  # Use axis coordinates
            )
            # # Inventory text
            # self.ax.text(
            #     0.575, -0.125,  
            #     'Inventory :',
            #     color='black',
            #     fontsize=caption_fontsize_small,
            #     ha='left',   
            #     transform=self.ax.transAxes  # Use axis coordinates
            # )
            # # Inventory value
            # self.ax.text(
            #     0.7, -0.125,
            #     f'light ({self.bc_inventory[FrameStructureType.FST_10_10]})     medium ({self.bc_inventory[FrameStructureType.FST_20_20]})',
            #     color='gray',
            #     fontsize=caption_fontsize_small,
            #     ha='left',   
            #     transform=self.ax.transAxes  # Use axis coordinates
            # )
        else:
            pass
            # print(f'Displacement is empty!')

        # # Interactive (debug_all Mode)
        # if self.render_mode == 'debug_all':
        #     # Ensure the canvas is available
        #     self.fig.canvas.draw()
        #     self.click_event_id = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        margin = 1
        self.ax.set_xlim([-margin, self.board_size_x + margin])
        self.ax.set_ylim([-margin, self.board_size_y + margin])
        self.ax.set_aspect('equal', adjustable='box')
        # self.ax.set_xticks(range(self.board_size_x + 1))
        # self.ax.set_yticks(range(self.board_size_y + 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

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
                render_path = os.path.join(self.render_dir, render_name)
                # Save the render
                self.render_frame()
                # plt.savefig(render_path, bbox_inches='tight')
                try:
                    plt.savefig(render_path, bbox_inches='tight')
                except Exception as e:
                    print(f"Error saving the plot to {render_path}: {e}")
                plt.close(self.fig)
                # Increment the counter for the next file
                self.render_counter += 1
        elif self.render_mode == "rgb_end_interval":
            render_name = f"render{self.render_counter}_eps{self.global_terminated_episodes}_step{self.global_steps}.png" 
            if self.eps_terminate_valid and self.global_terminated_episodes % self.render_interval_eps < self.render_interval_consecutive:
                render_path = os.path.join(self.render_dir, "img")
                os.makedirs(render_path, exist_ok=True) # make sure that render directory exists
                render_path = os.path.join(render_path, render_name)
                # Save the render
                self.render_frame()
                # plt.savefig(render_path, bbox_inches='tight')
                try:
                    plt.savefig(render_path, bbox_inches='tight')
                except Exception as e:
                    print(f"Error saving the plot to {render_path}: {e}")
                plt.close(self.fig)
                # Increment the counter for the next file
                self.render_counter += 1

        elif self.render_mode == "rgb_end_interval_nofail":
            render_name = f"render{self.render_counter}_eps{self.global_terminated_episodes}_step{self.global_steps}.png" 
            if self.eps_terminate_valid and self.global_terminated_episodes % self.render_interval_eps < self.render_interval_consecutive:
                if len(self.curr_fea_graph.failed_elements) == 0:
                    render_path = os.path.join(self.render_dir, "img")
                    os.makedirs(render_path, exist_ok=True) # make sure that render directory exists
                    render_path = os.path.join(render_path, render_name)
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
    def frame_to_board(self, frame_x, frame_y):
        '''
        Input
            (frame_x, frame_y) coordinates within frame grid
        Output
            board_coords : (x,y) centroid board coords of frame 
        '''
        board_x = frame_x*self.frame_size + self.frame_size//2
        board_y = frame_y*self.frame_size + self.frame_size//2
        return (board_x, board_y)
    
    def board_to_frame(self, center_board_x, center_board_y):
        '''
        Input
            center_board_x, center_board_y centroid board coords of frame
        Output
            frame_x, frame_y : frame grid coordinates
        '''
        # assert board coordinates are valid center of frame
        assert center_board_x % self.frame_size == self.frame_size // 2 and center_board_y % self.frame_size == self.frame_size // 2, \
        f"Invalid board coordinates: ({center_board_x}, {center_board_y}) are not the center of a frame"

        # convert board coordinates to frame coordinates
        frame_x = center_board_x // self.frame_size
        frame_y = center_board_y // self.frame_size
        return (frame_x, frame_y)

    

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
            self.human_action_frame_type = FrameStructureType.FST_10_10
        if event.key == '2':
            self.human_action_frame_type = FrameStructureType.FST_20_20
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
        Return
            action_mask : list of valid action tuples (end_bool, freeframe_idx, frame_x, frame_y)

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
        exists_end_action = False
        valid_actions = []
        for freeframe_type in FrameStructureType.get_free_frame_types():
            freeframe_idx_zero = freeframe_type.idx - 2 # convert FrameStructureType to start with index 0
            if self.inventory_dict[freeframe_type] > 0:
                for frame_x, frame_y in self.valid_pos:
                    # look ahead to check if support and target loads are connected
                    temp_is_connected = self.check_is_connected_bidirectional_temp(frame_x, frame_y) # dictionary of target center, boolean 

                    # with hypothetical frame, all targets are connected add only end action
                    if all(temp_is_connected.values()): # check if all values in the dictionary are true
                        # for end_bool in [False, True]:
                            # valid_actions.append((end_bool, freeframe_idx, frame_x, frame_y))
                        valid_actions.append((True, freeframe_idx_zero, frame_x, frame_y)) # only add action to end if connected
                        exists_end_action = True
                        valid_actions = [action for action in valid_actions if action[0] is True]
                    else:
                        if not exists_end_action: # Add action with end_bool=False only if no end_bool=True action exists
                            valid_actions.append((False, freeframe_idx_zero, frame_x, frame_y))


        if len(valid_actions) == 0:
            print(f'valid actions are empty!')
            print(f'inventory_dict : {self.inventory_dict}')
            print(f'valid_pos : {self.valid_pos}')
            self.print_framegrid()
            # print(f'temp_is_connected : {temp_is_connected}')
            return None

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
    def get_frame_grid_observation(self, condition=False):
        '''
        Used in step and reset to return frame grid observation
        condition = True (combine state with inventory condition) stack curr_frame_grid with current inventory
        '''

        if condition == True: # add inventory condition to observation vector
            # Get the number of columns in curr_frame_grid
            num_cols = self.curr_frame_grid.shape[0]
            inv_rows = []
            # Iterate over all frame types in the inventory dictionary
            num_free_frames = len(FrameStructureType.get_free_frame_types())
            for frame_type, inventory_value in self.inventory_dict.items():
                # Create a new row with zeros
                frame_row = np.zeros(num_cols, dtype=np.int64)
                # Fill the row with the inventory value for the current frame type
                frame_row[:inventory_value] = frame_type.idx + num_free_frames  # Assign unique values starting from last frame_type.idx
                # Expand dimensions to match the shape for stacking
                frame_row = np.expand_dims(frame_row, axis=-1)
                # Append the row to the list
                inv_rows.append(frame_row)

            # Stack all the rows with curr_frame_grid
            obs = np.hstack([self.curr_frame_grid] + inv_rows)  # Stack curr_frame_grid with all frame rows

        else:
            obs = self.curr_frame_grid.copy()  # Copy the current frame grid

        # print(f'stacked obs : {obs}')
        return obs
    
    # Render CSV mode
    def render_fixed_framegrid(self, fixed_framegrid):
        '''
        Used in render from csv mode to render unique designs 
        Input
            fixed_framegrid : fixed frame grid to render
        save rendered image to render_dir
        '''
        # get deflection of permuted frame grid
        self.curr_frame_grid = fixed_framegrid
        self.update_feagraph_from_framegrid() # update self.curr_fea_graph with new frame grid
        # perform fea
        self.update_displacement() # updates max deflection
        self.update_utilization()
        # print(f'###### rendering fixed framegrid fea graph ######: \n{self.curr_fea_graph}')
        self.eps_terminate_valid = True
        self.render()
        # print(f'max deflection : {self.max_deflection} at {self.max_deflection_node_idx}')
        # print(f'fea graph displacement : \n{self.curr_fea_graph.displacement}')
        # max_deflection_node_idx, self.max_deflection = self.curr_fea_graph.get_max_deflection()
        # store in csv file 
        # self.save_random_design_csv()
        self.global_terminated_episodes += 1

        # reset self.curr_frame_grid for next iteration
        self.curr_frame_grid = np.zeros((self.frame_grid_size_x, self.frame_grid_size_y), dtype=np.int64) # reset curr_frame_grid 
        self.frames = []
        self.initBoundaryConditions()

        gc.collect()

    # Functions to create baseline random samples
    def generate_random_designs(self, n_expand, n_permute):
        '''
        Used in baseline mode to generate random designs
        Create n random frame grids by 
            Generate random framegrids by 
            - create separate manhattan paths connecting support with all target loads
            - merge paths together
            - stochastically add adjacent frames * n times (at each round if not added, decrease probability)
            - randomly permute frame types and add framegrid to random framegrids
            Save to csv file (leaving fea data None)
        '''
        # Create 2 manhattan paths connecting support with all target loads
        all_paths = []
        support_board = [coord for tup in self.support_board for coord in tup] # unpack support_board list and tuple
        support_frame_coord = self.board_to_frame(*support_board) # convert board coords to frame coords
        for target_support_board_coord in self.target_support_board:
            target_support_frame_coord = self.board_to_frame(*target_support_board_coord) # convert board coords to frame coords
            # generate manhattan path between support and external load
            manhattan_path = self.find_manhattan_path(support_frame_coord, target_support_frame_coord)
            # add to list of manhattan paths
            if manhattan_path is not None:
                all_paths.append(manhattan_path)
        # merge paths to get all light frames
        all_frames = self.merge_manhattan_paths(all_paths)
        
        # add light frames to frame grid
        for frame_x, frame_y in all_frames:
            self.curr_frame_grid[frame_x, frame_y] = FrameStructureType.default_type.idx
        
        # Stochastically add adjacent frames * n_expand times
        frame_grid_prob = np.zeros(self.curr_frame_grid.shape, dtype=np.float32)
        for i in range(n_expand):
            valid_pos = self.get_valid_pos_design() # get all valid positions 
            # if valid_pos probability value is 0, set to 0.25
            valid_x_frame, valid_y_frame = zip(*valid_pos) # unpack valid_pos list of tuples
            frame_grid_prob[valid_x_frame, valid_y_frame] = 0.25
            # independently decide if each valid position is added
            random_values = np.random.rand(*self.curr_frame_grid.shape) # create array with self.curr_frame_grid.shape with random variables [0,1] for each cell

            add_mask = random_values < frame_grid_prob 
            self.curr_frame_grid[add_mask] = 2 # add light frames to frame grid
            frame_grid_prob[add_mask] = 0.0 # set probability to 0 for added frames

        print(f'number of frames after expand : {np.sum(self.curr_frame_grid == 2)}')

        # DEBUG generalize of all frame types
        path_frame_grid = np.array(copy.deepcopy(self.curr_frame_grid))
        for j in range(n_permute):
            # reset self.curr_frame_grid for next iteration
            self.curr_frame_grid = np.zeros((self.frame_grid_size_x, self.frame_grid_size_y), dtype=np.int64) # reset curr_frame_grid 
            # self.curr_fea_graph = FEAGraph()
            self.frames = []
            self.initBoundaryConditions()

            new_frame_grid = copy.deepcopy(path_frame_grid)
            # get positions where grid is filled with default frame
            def_pos = np.column_stack(np.where(new_frame_grid == FrameStructureType.default_type.idx))
            # shuffle positions to sample from
            np.random.shuffle(def_pos)
            available_pos = def_pos.tolist() # convert to list
            # for each frame type, take random number of positions to change in order from available_pos
            for frame_type in FrameStructureType.get_free_frame_types():
                # get number of frames to change
                max_replace = min(self.bc_inventory[frame_type], len(available_pos))
                if max_replace > 0:
                    # get random number of positions to change
                    rand_replace = np.random.randint(0, max_replace + 1)
                    # get random positions to change
                    for i in range(rand_replace):
                        row, col = available_pos.pop(0) # pop first element from available_pos
                        new_frame_grid[row, col] = frame_type.idx # change default frame to new frame type
            
            # get deflection of permuted frame grid
            self.curr_frame_grid = new_frame_grid
            self.update_feagraph_from_framegrid() # update self.curr_fea_graph with new frame grid
            # perform fea
            try:
                self.update_displacement() # updates max deflection
            except:
                print(f'update_displacement failed!')
                continue
            self.update_utilization()
            # print(f'###### Permutation {j} fea graph ######: \n{self.curr_fea_graph}')
            self.eps_terminate_valid = True
            self.render()
            # print(f'max deflection : {self.max_deflection} at {self.max_deflection_node_idx}')
            # print(f'fea graph displacement : \n{self.curr_fea_graph.displacement}')
            # max_deflection_node_idx, self.max_deflection = self.curr_fea_graph.get_max_deflection()
            # store in csv file 
            self.save_random_design_csv()
            self.global_terminated_episodes += 1



        gc.collect()

    def save_random_design_csv(self):
        
        boundary_condition = self.csv_bc # left height, left length, left magnitude, right height, right length, right magnitude
        inventory = self.csv_inventory # light, medium
        allowable_deflection = self.allowable_deflection
        episode_reward = self.episode_return
        terminated = True
        # Calculate or retrieve values for the current episode
        max_deflection = self.max_deflection
        utilization_min = self.utilization_min
        utilization_max = self.utilization_max
        utilization_median = self.utilization_median
        utilization_std = self.utilization_std
        utilization_percentile = self.utilization_ninety_percentile
        utilization_all_signed = [utilization for center_x, center_y, utilization, dir in self.edge_utilization]
        utilization_all = np.abs(utilization_all_signed) 
        num_frames = len(self.frames)
        num_failed = len(self.curr_fea_graph.failed_elements)
        frame_grid = self.curr_frame_grid

        with open(self.baseline_csv_path , mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)    
            # Write the row for the current episode
            csv_writer.writerow([
                self.global_terminated_episodes,
                terminated,
                boundary_condition,
                inventory,
                allowable_deflection,
                max_deflection,
                num_failed,
                utilization_min,
                utilization_max,
                utilization_median,
                utilization_std,
                utilization_percentile,
                utilization_all,
                num_frames,
                frame_grid,
                episode_reward,
            ])

    def get_valid_pos_design(self):
        '''
        used when self.valid_pos has not been updated at each step but only self.curr_frame_grid is updated
        get all valid positions based on current self.frame_grid
        '''
        # for all frames, 
        # get adjacent 4 cells
        # add if
        # within frame grid bounds
        # does not go beyond target cell x value
        # not existing frame 
        # does not go below support 
        valid_positions = []
        rows, cols = len(self.curr_frame_grid), len(self.curr_frame_grid[0])
        for i in range(rows):
            for j in range(cols):
                if self.curr_frame_grid[i][j] != FrameStructureType.UNOCCUPIED.idx and \
                    self.curr_frame_grid[i][j] != FrameStructureType.EXTERNAL_FORCE.idx:

                    # Check adjacent cells (up, down, left, right)
                    adjacent_cells = [
                        (i - 1, j),  # Up
                        (i + 1, j),  # Down
                        (i, j - 1),  # Left
                        (i, j + 1)   # Right
                    ]
                    for x_adj, y_adj in adjacent_cells:
                        # Check if the adjacent cell is within the frame grid bounds
                        if 0 <= x_adj < self.frame_grid_size_x and 0 <= y_adj < self.frame_grid_size_y:
                            # check if adjacent cell is in target x bounds
                            if self.target_x_bounds[0] <= x_adj and x_adj <= self.target_x_bounds[1]:
                                if self.curr_frame_grid[x_adj, y_adj] == FrameStructureType.UNOCCUPIED.idx or \
                                    self.curr_frame_grid[x_adj, y_adj] == FrameStructureType.EXTERNAL_FORCE.idx:
                                    # Add the valid position
                                    valid_positions.append((x_adj, y_adj))

        return valid_positions
    
    def merge_manhattan_paths(self, manhattan_paths):
        '''
        Merge multiple manhattan paths so there are no overlaps
        Returns single list of frame coordinates
        '''
        # Initialize an empty set to store unique coordinates
        merged_path = set()
        for path in manhattan_paths:
            for coord in path:
                # Add the coordinate to the set
                merged_path.add(coord)
        # Convert the set back to a list
        merged_path = list(merged_path)
        return merged_path

    def find_manhattan_path(self, start_frame_coords, end_frame_coords):
        '''
        Generate random manhattan path between start and end frame coordinates
        Return list of frame (x,y) coordinates excluding start and end
        '''
        if start_frame_coords == end_frame_coords:
            return None
        x_s, y_s = start_frame_coords
        x_e, y_e = end_frame_coords

        # Determine the steps in x and y directions
        steps = []
        if x_e > x_s:
            steps += ["R"] * (x_e - x_s)  # Right steps
        elif x_e < x_s:
            steps += ["L"] * (x_s - x_e)  # Left steps

        if y_e > y_s:
            steps += ["U"] * (y_e - y_s)  # Up steps
        elif y_e < y_s:
            steps += ["D"] * (y_s - y_e)  # Down steps

        # Shuffle the steps to create a random path
        random.shuffle(steps)

        # Generate the path coordinates
        path = []
        current_x, current_y = x_s, y_s
        for step in steps:
            if step == "R":
                current_x += 1
            elif step == "L":
                current_x -= 1
            elif step == "U":
                current_y += 1
            elif step == "D":
                current_y -= 1
            path.append((current_x, current_y))

        # Exclude the start and end coordinates
        return path[:-1] if path[-1] == end_frame_coords else path

    # Functions to handle predetermined frame grids
    def update_feagraph_from_framegrid(self):
        '''
        update self.feagraph with self.curr_frame_grid from predetermined frame grids 
        given that self.curr_frame_grid is initialized
        * at reset self.curr_fea_graph is initialized with edge_type_dict
        '''
        # Reset the Vertex ID counter
        Vertex._id_counter = 1

        self.curr_fea_graph = FEAGraph() # init new graph 
        self.initBoundaryConditions()

        # Add frames by their frame type index
        for frame_type in FrameStructureType.get_free_frame_types():
            # Find coordinates in the grid corresponding to the current frame type
            frame_coords = np.argwhere(self.curr_frame_grid == frame_type.idx)
            # Convert to board coordinates
            frame_board_coords = [(self.frame_to_board(x, y)) for x, y in frame_coords]
            for board_coord in frame_board_coords:
                # Add frame to FEA graph
                new_frame = TrussFrameRL(board_coord, type_structure=frame_type)
                self.update_fea_graph(new_frame)
                self.frames.append(new_frame)
                self.update_inventory_dict(new_frame)
        

