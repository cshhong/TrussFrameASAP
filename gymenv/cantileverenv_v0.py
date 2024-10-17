'''
Big Goals
TODO Random Rollout of environment
TODO Train with DQN 
TODO How many high performing solutions does the algo explore? 
TODO How many low performing solutions does the algo explore? 
    (aka how efficient is it solving the problem)
    - find efficient way to log solutions so far 
TODO Going further is there a way to get diverse high performing solutions across topologies instead of highest performing?
    (thresholding)
TODO Going further from general group of high performing designs -> perturb points -> how does performance change? 
    Get diverse group of high performing designs through refinement 

Smaller Goals
TODO env with supports created at end?
TODO Env - random placement of supports, target load 
        - connect all (as collection problem) -> FEA on final structure - likely will not be interesting solutions, make sure to make nodal weight heavy
        - connect all with limited options to add support? -> FEA on final structure

DONE step and reset
DONE render function 
TODO rollout with manual control?

DONE edit so that FEA is only run when end is indicated (otherwise no incentive to build after reaching target)
TODO reward : step reward to connect + end FEA reward 

DONE implement is valid action

TODO draw displacement at end
TODO clickable human mode


Resources
- Dynamic action masking : 
    
https://www.reddit.com/r/reinforcementlearning/comments/rlixoo/openai_gym_custom_environments_dynamically/
    -> use largest action space possible, use dummy values 
    -> give large negative reward to illegal action 
https://www.reddit.com/r/reinforcementlearning/comments/zj31h6/has_anyone_experience_usingimplementing_masking/
    -> changing the logits (distribution) but... 
<Termination vs Invalid actions>
    Effect on Training: Ignoring invalid actions might slow down learning since the agent isn't directly penalized for choosing them. 
    The agent might spend more time exploring invalid actions, leading to inefficient training. 
    However, it does avoid destabilizing the training process.
'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box, Dict, Graph
from gymnasium.spaces.graph import *

import sys
import os

# Add the directory containing TrussFrameMechanics to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TrussFrameMechanics.trussframe import FrameType, TrussFrameRL
from  TrussFrameMechanics.vertex import Vertex
from  TrussFrameMechanics.maximaledge import MaximalEdge
from  TrussFrameMechanics.feagraph import FEAGraph
import TrussFrameMechanics.generate_bc as generate_bc
import TrussFrameMechanics.pythonAsap_1 as pythonAsap_1

import juliacall
# from pythonAsap import solve_truss_from_graph

import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt


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
            'human' : occur during step, render, returns None
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
                - grid where each cell has value unoccupied= 0, free frame = 1, support frame = 2, force = -1
                
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
            upon taking step action is masked (invalid action has large neg reward) to cell adjacent of existing frame 
            end episode boolean indicates that the structure is complete (there is no deterministic end condition)
            
            1) Absolute Coordinates (end_bool, frame_x, frame_y)
            (decided not to use)
            2) Relative Coordinates (end_bool, frame_graph_id, left/right/top/bottom)
                - if there are multiple supports, should only be used with frame_graph state space

    
    '''
    metadata = {"render_modes": [None, "human", "rgb_list"], 
                "render_fps": 1,
                "obs_modes" : ['frame_grid', 'fea_graph', 'frame_graph'],
                }

    def __init__(self,
                 render_mode = None,
                 board_size_x=20,
                 board_size_y=10,
                 frame_size=2,
                 video_save_interval_steps=500,
                 render_dir = 'render',
                 max_episode_length = 20,
                 obs_mode='frame_grid'
                 ):
        
        self.board_size_x = board_size_x # likely divisable with self.frame_size
        self.board_size_y = board_size_y # likely divisable with self.frame_size
        self.frame_size = frame_size
        self.max_episode_length = max_episode_length
        self.env_num_steps = 0 # activated taking valid action in main (not env.step!)

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
        self.video_save_interval = video_save_interval_steps # in steps

        # Current State (Frames)
        self.frames=[] # stores structural frames in sequence of creation
        self.default_frame_type = FrameType.DIAGONAL_LT_RB
        self.curr_frame_type = self.default_frame_type
        # self.support_frames = [] # list of TrussFrameRL objects
        
        # Calculate the size of the frame grid based on the frame size
        self.frame_grid_size_x = self.board_size_x // self.frame_size
        self.frame_grid_size_y = self.board_size_y // self.frame_size

        # Initialize current state
        self.curr_frame_grid = np.zeros((self.frame_grid_size_x, self.frame_grid_size_y), dtype=int)
        
        self.curr_fea_graph = FEAGraph() #FEAGraph object
        self.curr_frame_graph = None # TODO graph representation of adjacent frames

        # if actions are taken outside of valid_pos, large negative reward is given.  
        self.valid_pos = set() # set of frame_grid coordinates (frame_x, frame_y) in which new frame can be placed 
        # if episode ends without being connected, large negative reward is given.
        self.target_loads_met = {} # Whether target load was reached : key is (x,y) coordinate on board value is True/False
        self.is_connected = False # whether the support is connected to (all) the target loads, 

        # Set current observation based on observation mode
        self.obs_mode = obs_mode  # Set the observation mode
        if self.obs_mode not in self.metadata["obs_modes"]:
            raise ValueError(f"Invalid observation mode: {self.obs_mode}. Valid modes are: {self.metadata['obs_modes']}")
        self.curr_obs = None
        self._update_curr_obs() # Set according to obs_mode

        self.observation_space = None
        self.action_space = None
        self._set_gym_spaces()

        # self.initBoundaryConditions() # TODO is this necessary? 

        print("Initialized Cantilever Env!")

    def reset(self, seed=None, **kwargs):
        '''
        Create boundary condition within environment with 
            generate_bc.set_cantilever_env_framegrid(self.frame_grid_size_x)
        that returns support_frames, targetload_frames within the frame grid
        self.frames, self.valid_pos, self.curr_frame_grid, self.curr_frame_graph, self.curr_fea_graph is updated

        '''
        print('Resetting Env!')

        self.render_list = []
        self.env_num_steps = 0
        
        self.frames = []
        self.curr_frame_type = self.default_frame_type

        # Reset the current state
        self.curr_frame_grid = np.zeros((self.frame_grid_size_x, self.frame_grid_size_y), dtype=int)
        self.curr_fea_graph = FEAGraph() #FEAGraph object
        self.curr_frame_graph = None # TODO graph representation of adjacent frames
        self.valid_pos = set()

        self._update_curr_obs() 

        # Set boundary conditions
        self.initBoundaryConditions()
        print(f'target loads met : {self.target_loads_met}')

        obs = self.curr_obs
        info = {} # TODO what is this used for?
        
        self.render()
        print(f"valid pos : {self.valid_pos}")

        return obs, info
    
    def step(self, action):
        '''
        Accepts an action, computes the state, reward of the environment after applying that action 
        and returns the 5-tuple (observation, reward, terminated, truncated, info).
        Action is (end_bool, frame_x, frame_y) coordinate chosen by agent
        If action is invalid, produce large negative reaction &/ terminate
            registers transition (s=curr_state, a=action, s_prime=curr_state, r=-10, truncated=False)
            In theory with termination agent may learn how to connect faster?
            But also not terminating creates more valid transitions
        
        Input 
            action : (end_bool, frame_x, frame_y) 
                
        Returns:
            observation, reward, terminated, truncated, info
        '''
        # Large negative reward is given if action taken is not in valid position
        end, frame_x, frame_y = action
        end_bool = True if end==1 else False
        
        # Large negative reward is given if position is not valid
        if (frame_x, frame_y) not in self.valid_pos:
            reward = -10  # Negative reward for invalid action
            terminated = False  # Continue episode
            # Do not apply this action to the environment!
        else:
            temp_is_connected = self._check_is_connected(frame_x, frame_y)
            # Large negative reward is given if decide to end episode but support and target not connected
            # (action is not applied to environment)
            if end_bool == True and temp_is_connected == False:
                reward = -10  # Negative reward for invalid action
                terminated = False  # Continue episode
            # Correctly ended after support and loads are all connected
            elif end_bool == True and temp_is_connected == True:
                # Apply valid action and update environment state
                self.apply_action(action)
                # TODO add displacement reward
                reward = 10 # TODO get final structural reward (Big pos - negative for element count ) 
                self.draw_truss_analysis(self) # last plot has displaced structure 
                terminated = True
            elif end_bool == False:
                self.apply_action(action)
                reward = 1 # small reward for creating block 
                terminated = False

        obs = self.curr_obs  # New observation after applying the action
        print(f"current FEAGraph : \n {self.curr_fea_graph} ")
        self.print_framegrid()
        truncated = False  # Assuming truncation is handled elsewhere or is not used

        # is render() is triggered automatically at each step?
        # Display render every step for 'human' mode TODO why not for 'rbg_list' mode??
        self.render()

        return obs, reward, terminated, truncated, {}
    
    def _check_is_connected(self, frame_x, frame_y):
        '''
        Input
            frame center coordinates (frame_x, frame_y)
        Check if overall structure is connected (support and target load)
        Assumption that frames can only be built in adjacent cells allows us to check only most recent frame
        Does temporary forward checking (this frame is not necessarily created in env)
        '''
        # check if top-right, or top-left node of frame changes current self.target_loads_met values
        # given temporary changed values, if all are true, return True
        temp_target_loads_met = self.target_loads_met.copy()
        
        center = self._framegrid_to_board(frame_x, frame_y) # get center board coordinates
        top_right = (center[0] + self.frame_size//2, center[1] + self.frame_size//2)
        top_left = (center[0] - self.frame_size//2, center[1] + self.frame_size//2)

        for target in self.target_loads_met:
            if top_right == target:
                temp_target_loads_met[target] = True
            if top_left == target:
                temp_target_loads_met[target] = True

        print(f"_check_is_connected : {temp_target_loads_met}")
        
        # Check if all target loads are met
        if all(temp_target_loads_met.values()):
            return True
        else:
            return False

    
    def apply_action(self, valid_action):
        '''
        Used in step to apply action to current state
        Assumed that valid action has been checked, thus only used with valid actions
        Input 
            valid_action : (end_bool, frame_x, frame_y) coordinate 
        Updates frame_grid, fea_graph, curr_obs, frames, target_load_met and is_connected
        '''
        # create free TrussFrameRL at valid_action board coordinate
        end, frame_x, frame_y = valid_action
        frame_center = self._framegrid_to_board(frame_x, frame_y)
        new_frame = TrussFrameRL(pos = frame_center)
        
        # update current state 
        self.update_frame_grid(new_frame)
        self.update_fea_graph(new_frame)
        if end == 1: # update displacement info in fea graph if episode end
            self.update_displacement()
        # TODO self.update_frame_graph(new_frame)

        self._update_curr_obs()

        self.frames.append(new_frame)

        self.update_target_meet(new_frame)

    def draw_fea_graph(self):
        '''
        Update self.fig and self.ax based on self.curr_fea_graph
        used in _render_frame
        '''
        text_offset = 0.1

        vertices = self.curr_fea_graph.vertices.items() # coord, Vertex object pairs
        maximal_edges = self.curr_fea_graph.maximal_edges.items()
        supports = self.curr_fea_graph.supports # list of board coords where the nodes are supports / pinned (as opposed to free)
        # Draw vertices
        for coord, vertex in vertices:
            # if vertex.coordinates in supports:
            if vertex.is_free == False:
                self.ax.add_patch(patches.Rectangle((coord[0] - 0.1, coord[1] - 0.1), 0.2, 0.2, color='blue', lw=1.5, fill=True))
            else:
                self.ax.add_patch(patches.Circle(coord, radius=0.1, color='blue', lw=1.5, fill=False ))
            self.ax.text(coord[0]-text_offset, coord[1]+text_offset, 
                         str(vertex.id), 
                         fontsize=10, ha='right', color='black')

        # Draw maximal edges (optional, if visually distinct from normal edges)
        for direction, edges in maximal_edges:
            for edge in edges:
                if len(edge.vertices) >= 2:
                    # Get start and end vertices from the list of vertices
                    start_me = edge.vertices[0]
                    end_me = edge.vertices[-1]
                    
                    # Draw the line connecting the start and end vertices
                    self.ax.plot([start_me.coordinates[0], end_me.coordinates[0]], 
                                [start_me.coordinates[1], end_me.coordinates[1]], 
                                color='black', linestyle='-', linewidth=1)
                else:
                    print(f"Warning: Maximal edge in direction {direction} has less than 2 vertices and cannot be drawn.")
        
        # Draw external forces as red arrows
        for coord, load in self.curr_fea_graph.external_loads.items():
            force_magnitude = (load[0]**2 + load[1]**2 + load[2]**2)**0.5
            if force_magnitude > 0:
                arrow_dx = load[0] * 0.025
                arrow_dy = load[1] * 0.025
                arrow_tail_x = coord[0] - arrow_dx
                arrow_tail_y = coord[1] - arrow_dy
                # arrow_head_x = arrow_tail_x - arrow_dx
                # arrow_head_y = arrow_tail_y - arrow_dy

                self.ax.arrow(arrow_tail_x, arrow_tail_y, arrow_dx, arrow_dy+0.1, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
                self.ax.text(arrow_tail_x, arrow_tail_y + 0.1, f"{force_magnitude:.2f} kN", color='red', fontsize=10, fontweight='bold')

    def _render_frame(self):
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
        self.ax.set_xticks(range(self.board_size_x + 1))
        self.ax.set_yticks(range(self.board_size_y + 1))

        # Draw grid lines
        self.ax.grid(True, which='both', color='lightblue', linestyle='-', linewidth=0.5,  zorder=0)
        for i in range(0, self.board_size_x + 1, 2):
            self.ax.axvline(x=i, color='lightblue', linestyle='-', linewidth=2, zorder=0)
        for j in range(0, self.board_size_y + 1, 2):
            self.ax.axhline(y=j, color='lightblue', linestyle='-', linewidth=2, zorder=0)

        # Highlight valid position cells
        for frame_x, frame_y in self.valid_pos:
            x , y = self._framegrid_to_board(frame_x, frame_y)
            rect = patches.Rectangle((x - self.frame_size//2, y - self.frame_size//2), 
                                     self.frame_size, self.frame_size, 
                                     linewidth=0, edgecolor='lightblue', facecolor='lightblue')
            self.ax.add_patch(rect)
        
        # Draw current fea graph
        self.draw_fea_graph() # update self.fig, self.ax with current fea graph 

        # Overlap deflected truss 
        self.draw_truss_analysis()


    def render(self):
        '''
        Given that we are working with from gymnasium.utils.save_video import save_video 
        Trigger final render action. Used in main loop
            - 'rgb_list' : return fig in rgb format to be saved in RenderList Wrapper item 
            - 'human' : display plot
        '''
        self._render_frame() # initialize and update self.ax, self.fig object
        if self.render_mode == "rgb_list":
            img = self._fig_to_rgb_array(self.fig)
            plt.close(self.fig)
            # print(f'returning img for rgb_array!')
            return img
        elif self.render_mode == "human":
            plt.show()
            plt.close(self.fig)
        else:
            raise NotImplementedError(f"Render mode {self.render_mode} is not supported")
        
    def _fig_to_rgb_array(self, fig):
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


    def initBoundaryConditions(self):
        support_frames, targetload_frames = generate_bc.set_cantilever_env_framegrid(self.frame_grid_size_x) # within frame grid
        print(f"    support_frames : {support_frames}")
        print(f"    targetload_frames : {targetload_frames}")

        # init supports in curr_frame_grid according to bc
        for s_frame_coords in support_frames:
            s_board_coords = self._framegrid_to_board(*s_frame_coords) #convert from frame grid coords to board coords
            new_s_frame = TrussFrameRL(s_board_coords, type_structure=2)
            self.frames.append(new_s_frame)
            self.update_frame_grid(new_s_frame)
            self.update_frame_graph(new_s_frame)
            self.update_fea_graph(new_s_frame)

        for t_frame in targetload_frames.items():
            t_frame_coord, t_load_mag = t_frame # (x,y) on frame grid, magnitude in kN
            # convert from frame grid coords to board coords 
            t_center_board = self._framegrid_to_board(*t_frame_coord) # center of proxy frame
            t_load_board = (t_center_board[0]+self.frame_size//2 , t_center_board[1]-self.frame_size//2)# actual load board coordinate (bottom right of frame)
            # t_board_coord = (self._framegrid_to_board(t_frame_coord)[0] + self.frame_size//2, self._framegrid_to_board(t_frame_coord)[1] - self.frame_size//2) 
            new_t_frame = TrussFrameRL(t_center_board, type_structure=-1)
            # self.frames.append(new_t_frame)
            self.update_frame_grid(new_t_frame)
            self.update_frame_graph(new_t_frame)
            self.update_fea_graph(new_t_frame, t_load_mag) #TODO need implementation
            
            self.target_loads_met[t_load_board] = False

    def draw_truss_analysis(self):
        '''
        Given displacement for a graph, plot graph, nodal displacement, and max displacement
        '''
    
    def update_displacement(self):
        '''
        Used in apply_action 
        Called upon action that indicates end of episode (end_bool == 1)
        Performs FEA with Julia ASAP 
        Updates self.curr_fea_graph displacement attribute
        '''

        # Solve truss model with ASAP
        jl = juliacall.newmodule("TrussFrameRL") 
        curr_env = jl.seval('Base.active_project()')
        print(f"The current active Julia environment is located at: {curr_env}")

        # Step 0: Initialize Julia session and import necessary Julia modules
        jl.seval('using AsapToolkit')
        jl.seval('using Asap')

        jl.include("TrussFrameMechanics/truss_analysis.jl")
        jl.seval('using .TrussAnalysis')

        displacement = pythonAsap_1.solve_fea(jl, self.curr_fea_graph) # return nodal displacement
        self.curr_fea_graph.displacement = displacement
        
    
    def _update_curr_obs(self):
        '''
        Updates self.curr_obs in place the current observation based on the selected mode.
        Make sure to use after updating states of different modes
        '''
        if self.obs_mode == 'frame_grid':
            self.curr_obs = self.curr_frame_grid
        elif self.obs_mode == 'fea_graph':
            self.curr_obs = self.curr_fea_graph.get_state()
        elif self.obs_mode == 'frame_graph':
            self.curr_obs = self.curr_frame_graph.get_state() if self.curr_frame_graph else None
        else:
            raise ValueError(f"Invalid observation mode: {self.obs_mode}")
        
    def _set_gym_spaces(self):
        '''
        set observation_space, action_space according to obs mode
        '''
        # Set observation_space and action_space (following Gymnasium)
        if self.obs_mode == 'frame_grid':
            # Define Observations : frame grid 
            #   - array shape (frame_grid_size_x, frame_grid_size) with int values [-1,2]
            self.observation_space = Box(low=-1, high=2, shape=(self.frame_grid_size_x, self.frame_grid_size_y), dtype=np.int32) 
            print(f'Total Number of Possible States: > {2**(self.frame_grid_size_x * self.frame_grid_size_y)}')
            # Define Actions : end boolean, absolute coordinates 
            #   - tuple (end_bool, frame_x, frame_y) with int values [0,1] , [0, self.frame_grid_size_x], [0, self.frame_grid_size_y]
            self.action_space = Box(low = np.array([0, 0, 0]), 
                                    high= np.array([1, self.frame_grid_size_x, self.frame_grid_size_y]),
                                    dtype=np.int32)
            print(f'Total Number of Actions : {2*self.frame_grid_size_x*self.frame_grid_size_y}') # 2*10*5

            # Relative Coordinates (end_bool, frame_idx, left/right/top/bottom) ---> Doesn't save much from absolute coordinates, still have to check valid action!
            # self.action_space = Box(low = [0, 0, 0], 
            #                         high= [1, self.max_episode_length, 3],
            #                         shape=(1, 1, 1), dtype=np.int32)
            # print(f'Total Number of Actions : < {2*self.max_episode_length*4}') #2*20*4

        elif self.obs_mode == 'fea_graph':
            print('TODO Need to implement _set_gym_spaces for fea_graph!')
            pass
            # Gymnasium Composite Spaces - Graph or Dict?
            # Graph - node_features, edge_features, edge_links
            # Dict (not directly used in learning but can store human interpretable info)
        elif self.obs_mode == 'frame_graph':
            print('TODO Need to implement _set_gym_spaces for frame_graph!')
            pass


    def update_frame_grid(self, new_frame):
        '''
        Given new frame object, update current frame grid where 
        Input
            TrussFrameRL object 
        
        Updates:
        - self.curr_frame_grid : A grid where each cell is updated based on the frame type.
            - (cell state for array size frame_grid_size_x frame_grid_size_y) 
            - grid where each cell has value unoccupied= 0, free frame = 1, support frame = 2, force = -1
        - self.valid_pos : A set of valid (x, y) frame positions on the frame grid where a new frame can be placed.
        '''
        # Update the current frame grid with the new frame's type
        self.curr_frame_grid[new_frame.x_frame, new_frame.y_frame] = new_frame.type_structure

        # Remove the position of the new frame from valid_pos if it exists
        if (new_frame.x_frame, new_frame.y_frame) in self.valid_pos:
            self.valid_pos.remove((new_frame.x_frame, new_frame.y_frame))
        # else:
        #     raise ValueError(f"Position ({new_frame.x_frame}, {new_frame.y_frame}) is not a valid position for placing a frame.")

        # Update valid position if frame not load frame
        if new_frame.type_structure != -1: 
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
                    if self.curr_frame_grid[x_adj, y_adj] == 0 or self.curr_frame_grid[x_adj, y_adj] == -1:
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

        vertices = self.curr_fea_graph.vertices # dictionary of vertices with coordinate as key and Vertex object as value
        def_load = self.curr_fea_graph.default_node_load

        if new_frame.type_structure == -1: # target frame
            # Bottom right vertex of proxy frame is registered as load but not as Vertex object
            target_load_pos = vert_pos[1]
            self.curr_fea_graph.external_loads[target_load_pos] = t_load_mag
            # TODO need to cross check with existing Vertices if target load is added after environment init
        else: # free or support
            new_vertices = [] # Vertex object in order of bottom-left, bottom-right, top-right, top-left
            for i, pos in enumerate(vert_pos):
                # If new node overlaps with existing node, merge (preserve existing node attributes - id, is_free)
                if pos in vertices:
                    new_v = vertices[pos] # get overlapping existing node
                    # allow change free->fixed but not fixed->free
                    if new_frame.type_structure == 2:
                        new_v.is_free = False
                # If node does not overlap with existing node, create new node
                else: 
                    if new_frame.type_structure == 1: # Free
                        new_v = Vertex(pos, is_free=True, load=def_load)
                    elif new_frame.type_structure == 2: # Support
                        if i==0 or i==1: # Bottom left, Bottom right are fixed 
                            new_v = Vertex(pos, is_free=False, load=def_load)
                            self.curr_fea_graph.supports.append(pos) # add to list of supports
                        else:
                            new_v = Vertex(pos, is_free=True, load=def_load)
                    
                    # additionally check if meets with external load
                    if pos in self.curr_fea_graph.external_loads:
                        # new_v.load += self.curr_fea_graph.external_loads[pos]
                        new_v.load = [x + y for x, y in zip(new_v.load, self.curr_fea_graph.external_loads[pos])]

                    # add new node to fea graph
                    vertices[pos] = new_v 
                # add to new vertices to combine edges                    
                new_vertices.append(new_v) 

            # Check line overlap with existing edge using maximal edge representation 
            self.curr_fea_graph.combine_and_merge_edges(frame_type_shape=new_frame.type_shape, new_vertices=new_vertices)


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

    def _framegrid_to_board(self, frame_x, frame_y):
        '''
        Input
            (frame_x, frame_y) coordinates within frame grid
        Output
            board_coords : (x,y) centroid board coords of frame 
        '''
        board_x = frame_x*self.frame_size + self.frame_size//2
        board_y = frame_y*self.frame_size + self.frame_size//2
        return (board_x, board_y)
    
    def print_framegrid(self):
        '''
        Prints the current frame grid in a human-readable format with the x-axis across different lines.
        '''
        print("Current Frame Grid:")
        transposed_grid = self.curr_frame_grid.T  # Transpose the grid
        for row in reversed(transposed_grid):
            print(" ".join(map(str, row)))