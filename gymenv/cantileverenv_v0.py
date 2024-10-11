'''
TODO env with supports created at end?
TODO Env - random placement of supports, target load 
        - connect all (as collection problem) -> FEA on final structure - likely will not be interesting solutions, make sure to make nodal weight heavy
        - connect all with limited options to add support? -> FEA on final structure

TODO render function 
TODO rollout with manual control?

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

from TrussFrameASAP.trussframe import FrameType, TrussFrameRL
from  TrussFrameASAP.vertex import Vertex
from  TrussFrameASAP.maximaledge import MaximalEdge
from  TrussFrameASAP.feagraph import FEAGraph
import TrussFrameASAP.generate_env as generate_env


import numpy as np
import torch
import copy

import os
import gc

import random

class CantileverEnv_0(gym.Env, obs_mode='frame_grid'):
    '''
        use - gymnasium.make("CartPole-v1", render_mode="human")

        Initialize the environment with a specified observation mode.
        Observation Modes: 
        - 'frame_grid': Only use the frame grid.
        - 'fea_graph': Only use the FEA graph.
        - 'frame_graph': Only use the frame graph.
        
        Render modes:
            human : occur during step, render, returns None
            rgb  : Return a single plot images representing the current state of the environment.
                   A plot is created by matplotlib
            rgb_list :  Returns a list of plots

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

        Action : (end episode boolean, relative_x, relative_y)
            create frame within framegrid masked to cell adjacent of existing frame 
            coordinates are relative to start support 
            end episode boolean indicates that the structure is complete (there is no deterministic end condition)
    
    '''
    metadata = {"render_modes": ["human", "rgb", "rgb_list"], "render_fps": 1}

    def __init__(self,
                 render_mode = None,
                 board_size_x=20,
                 board_size_y=10,
                 frame_size=2,
                 video_save_interval_steps=500,
                 max_episode_length = 20,
                 obs_mode='frame_grid'
                 ):
        
        self.board_size_x = board_size_x # likely divisable with self.frame_size
        self.board_size_y = board_size_y # likely divisable with self.frame_size
        self.frame_size = frame_size
        self.max_episode_length = max_episode_length
        self.env_num_steps = 0 # activated taking valid action in main (not env.step!)

        # Render
        self.fig = None
        # self.figsize = 
        self.ax = None
        self.render_size = (15, 8)
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_list = [] # gathers plts to save to results in "rgb_list" mode
        self.video_save_interval = video_save_interval_steps # in steps

        # Current State (Frames)
        self.frames=[] # stores frames in sequence of creation
        self.default_frame_type = FrameType.DIAGONAL_LT_RB
        self.curr_frame_type = self.default_frame_type
        # self.support_frames = [] # list of TrussFrameRL objects
        
        # Calculate the size of the frame grid based on the frame size
        self.frame_grid_size_x = self.board_size_x // self.frame_size
        self.frame_grid_size_y = self.board_size_y // self.frame_size

        # Initialize current state
        self.curr_frame_grid = np.zeros((self.frame_grid_size_y, self.frame_grid_size_x), dtype=int)
        self.curr_fea_graph = FEAGraph() #FEAGraph object
        self.curr_frame_graph = None # TODO graph representation of adjacent frames
        self.valid_pos = set() # set of board pos (x,y) in which new frame can be placed 

        self.obs_mode = obs_mode  # Set the observation mode
        self.curr_obs = None # Set according to obs_mode

        self.initBoundaryConditions() # TODO is this necessary? 

        # TODO Visualize board as frame_grid, fea_graph (Visualizing representations)

        print("Initialized Cantilever Env!")

    def reset(self, seed=None, **kwargs):
        '''
        Create boundary condition within environment with 
            generate_env.set_cantilever_env_framegrid(self.frame_grid_size_x)
        that returns support_frames, targetload_frames within the frame grid
        self.frames, self.valid_pos, self.curr_frame_grid, self.curr_frame_graph, self.curr_fea_graph is updated

        '''
        print('Resetting Env!')

        self.render_list = []
        self.env_num_steps = 0
        
        self.frames = []
        self.curr_frame_type = self.default_frame_type

        # Reset the current state
        self.curr_frame_grid = np.zeros((self.frame_grid_size_y, self.frame_grid_size_x), dtype=int)
        self.curr_fea_graph = FEAGraph() #FEAGraph object
        self.curr_frame_graph = None # TODO graph representation of adjacent frames
        self.valid_pos = set()

        self.curr_obs = self._update_curr_obs()

        # Set boundary conditions
        self.initBoundaryConditions()

        obs = self.curr_frame_grid 
        info = None # TODO what is this used for?

        return obs, info
    
    def step(self, action):
        '''
        Accepts an action, computes the state, reward of the environment after applying that action 
        and returns the 5-tuple (observation, reward, terminated, truncated, info).
        Action is (frame_x, frame_y) coordinate chosen by agent
        If action is invalid, produce large negative reaction &/ terminate
            registers transition (s=curr_state, a=action, s_prime=curr_state, r=-10, truncated=False)
            In theory with termination agent may learn how to connect faster?
            But also not terminating creates more valid transitions
        
        Input 
            action : (frame_x, frame_y) 
                
        Returns:
            observation, reward, terminated, truncated, info
        '''
        # Check if action is valid
        if not self.is_valid_action(action):
            reward = -10  # Negative reward for invalid action
            terminated = False  # Continue episode
            obs = self.curr_obs()  # Return the current state
        else:
            # Apply valid action and update environment state
            self.apply_action(action)
            reward = self.compute_reward() # based on changed state
            obs = self.curr_obs()  # New observation after applying the action
            terminated = self.check_termination()
        truncated = False  # Assuming truncation is handled elsewhere or is not used

        return obs, reward, terminated, truncated, {}
    
    def apply_action(self, valid_action):
        '''
        Apply action to current state
        Assumed that valid action has been checked, thus only used with valid actions
        Input 
            valid_action : (frame_x, frame_y) coordinate 
        '''
        # create free TrussFrameRL at valid_action board coordinate
        frame_center = self._framegrid_to_board(valid_action)
        new_frame = TrussFrameRL(pos = frame_center)
        
        # update current state 
        self.update_frame_grid(new_frame)
        self.update_fea_graph(new_frame)
        # TODO self.update_frame_graph(new_frame)

        self._update_curr_obs()

        self.frames.append(new_frame)


    def initBoundaryConditions(self):
        support_frames, targetload_frames = generate_env.set_cantilever_env_framegrid(self.frame_grid_size_x) # within frame grid

        # init supports in curr_frame_grid according to bc
        for s_frame_coords in support_frames:
            s_board_coords = self._framegrid_to_board(s_frame_coords) #convert from frame grid coords to board coords
            new_s_frame = TrussFrameRL(s_board_coords, type=2)
            self.frames.append(new_s_frame)
            self.update_frame_grid(new_s_frame)
            self.update_frame_graph(new_s_frame)
            self.update_fea_graph(new_s_frame)

        for t_frame in targetload_frames:
            t_frame_coord, t_load_mag = t_frame
            # convert from frame grid coords to board coords 
            t_board_coord = self._framegrid_to_board(t_frame_coord) # center of proxy frame
            # t_board_coord = (self._framegrid_to_board(t_frame_coord)[0] + self.frame_size//2, self._framegrid_to_board(t_frame_coord)[1] - self.frame_size//2) 
            new_t_frame = TrussFrameRL(t_board_coord, type=-1)
            self.frames.append(new_t_frame)
            self.update_frame_grid(new_t_frame)
            self.update_frame_graph(new_t_frame)
            self.update_fea_graph(new_t_frame, t_load_mag)

    def compute_reward(self):
        '''
        Compute reward based on current state
        Make sure that action is taken before computing reward
        '''
        pass
    
    def _update_curr_obs(self):
        '''
        Update the current observation based on the selected mode.
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
        else:
            raise ValueError(f"Position ({new_frame.x_frame}, {new_frame.y_frame}) is not a valid position for placing a frame.")

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
                # If the adjacent cell is unoccupied, add it to valid_pos
                if self.curr_frame_grid[x_adj, y_adj] == 0:
                    if (x_adj, y_adj) not in self.valid_pos:
                        self.valid_pos.add((x_adj, y_adj)) # 

    def update_frame_graph(self, new_frame):
        '''
        Given new TrussFrameRL object that is placed, update current frame graph where 
            - nodes are TrussFrameRL objects
            - edges are physical adjacencies, relative distance to external forces and supports
        '''
        pass #TODO
    
    def update_fea_graph(self, new_frame, t_load_mag=[0,0,0]):
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
        half_size = self.square_size / 2
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
                if new_frame.type_structure == 1:
                    is_free = True
                elif new_frame.type_structure == 2:
                    if i==0 or i==1: # Bottom left, Bottom right
                        is_free = False
                    else:
                        is_free = True
                if pos in vertices: # node exists in current fea graph merge overlapping new nodes with existing nodes
                    new_v = vertices[pos]
                    new_v.is_free = is_free # DEBUG change existing node is free property
                else: # If the node does not exist in the current graph, create new node
                    new_v = Vertex(pos, is_free=is_free, load=def_load)
                    if pos in self.curr_fea_graph.external_loads:
                        self._add_external_load(new_v)
                    vertices[pos] = new_v # add node to fea graph
                new_vertices.append(new_v) 

            # Check line overlap with existing edge using maximal edge representation 
            self.curr_fea_graph.combine_and_merge_edges(frame_type=self.curr_frame_type, new_vertices=new_vertices)


    def _framegrid_to_board(self, framegrid_coords):
        '''
        Input
            framegrid_coords : (x_frame, y_frame) coordinates within frame grid
        Output
            board_coords : (x,y) centroid board coords of frame 
        '''
        board_x = framegrid_coords[0]*self.frame_size + self.frame_size//2
        board_y = framegrid_coords[1]*self.frame_size + self.frame_size//2
        return (board_x, board_y)
        