'''
TODO env with supports created at end

'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box, Dict, Graph
from gymnasium.spaces.graph import *

import TrussFrameASAP.generate_env
from TrussFrameASAP.trussframe import FrameType, TrussFrameRL
from  TrussFrameASAP.vertex import Vertex
from  TrussFrameASAP.maximaledge import MaximalEdge
from  TrussFrameASAP.feagraph import FEAGraph


import numpy as np
import torch
import copy

import os
import gc

import random

class CantileverEnv_0(gym.Env):
    '''
        use - gymnasium.make("CartPole-v1", render_mode="human")
        Render modes:
            human : occur during step, render, returns None
            rgb  : Return a single plot images representing the current state of the environment.
                   A plot is created by matplotlib
            rgb_list :  Returns a list of plots

        State / Observation : 
            board 
                - (cell state for array size board_size_x, board_size_y) 
                - board grid where each cell has value unoccupied= 0, free frame = 1, support frame = 2, force = -1
            frame_graph 
                - nodes represent frames and edges represent adjacency, and relative distance to support/external force

        Action : (end episode boolean, relative_x, relative_y)
            create frame (constrainted to cell adjacent of existing frame) : (x,y) coordinate relative to start support 
            end episode 
    
    '''
    metadata = {"render_modes": ["human", "rgb", "rgb_list"], "render_fps": 1}

    def __init__(self,
                 render_mode = None,
                 board_size_x=20,
                 board_size_y=10,
                 cell_size=1,
                 frame_size=2,
                 video_save_interval_steps=500,
                 max_episode_length = 20,
                 ):
        
        self.board_size_x = board_size_x
        self.board_size_y = board_size_y
        self.cell_size = cell_size
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
        self.support = [] # list of TrussFrameRL objects
        self.external_force = [] # list of ? objects
        
        self.curr_board = np.zeros(self.board_size_x, self.board_size_y) # TODO np.array?
        self.curr_fea_graph = None #FEAGraph object
        self.curr_frame_graph = None # TODO graph representation of adjacent frames
        self.valid_pos = [] # list of board pos (x,y) in which new frame can be placed 

        # Mask actions
        # self.action_maps = self.get_all_actions()
        # self.initial_state = self.curr_state.graph # Used for replaybuffer

        # Handle overlap and connecting elements for Frame -> Structural Model node, edges
        # TODO Handle within fea graph!
        self.vertices = dict() # dictionary of vertices with coordinate as key and Vertex object as value
        self.edges = [] # adjacency list of tuples of vertex indices pairs
        self.maximal_edges = {
            'horizontal': [],
            'vertical': [],
            'LB_RT': [], 
            'LT_RB': []
        }

        # Boundary Conditions 
        self.per_node_load = -0.4
        default_frames, supports, target_load = TrussFrameASAP.generate_env.set_cantilever_env(self.board_size_x, self.square_size, seed=seed)
        print(f'default_frames : {default_frames}, supports : {supports}, target load : {target_load}')
        self.target_loads = target_load 
        self.supports = supports # dictionary of vertex.id : grid coordinates that are designated as supports / pinned as opposed to free
        
        # Init frames according to BC 
        for f_coords in default_frames:
            new_frame = TrussFrameRL(f_coords, support=True)
            self.frames.append(new_frame)
            self.update_valid_pos(new_frame)
            self.update_curr_board(new_frame)
            self.update_curr_frame_graph(new_frame)
            # self.update_curr_fea_graph(new_frame)

            

        print("Initialized Cantilever Env!")

    def reset(self, seed=None, **kwargs):
        print('Resetting Env!')

        self.render_list = []
        self.env_num_steps = 0
        
        self.frames = []
        self.curr_frame_type = self.default_frame_type


    
    def update_valid_pos(self, new_frame):
        '''
        Given new TrussFrameRL object, update self.valid_pos (list of (x,y) where new frame can be placed) 
        '''
        pass #TODO

    def update_curr_board(self, new_frame):
        '''
        Given new TrussFrameRL object that is placed, update current board where 
        board 
            - (cell state for array size board_size_x, board_size_y) 
            - grid where each cell has value unoccupied= 0, free frame = 1, support frame = 2, force = -1
        '''
        pass #TODO

    def update_curr_frame_graph(self, new_frame):
        '''
        Given new TrussFrameRL object that is placed, update current frame graph where 
            - nodes are TrussFrameRL objects
            - edges are physical adjacencies, relative distance to external forces and supports
        '''
        pass #TODO
    
    def update_curr_fea_graph(self, new_frame):
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

        # Merge overlapping new nodes with existing nodes
        new_vertices = [] # Vertex object in order of bottom-left, bottom-right, top-right, top-left
        for pos in vert_pos:
            # get node index if node exists in self.nodes
            if pos in vertices:
                new_v = vertices[pos]
            else: # If the node does not exist in the current graph, create new node
                vert_idx = len(vertices)  # Assign a new index to the node
                new_v = Vertex(pos)
                self._update_load(new_v) # update load 
                assert len(vertices)+1 == new_v.id, f'node index mismatch between {len(self.vertices)} and {vert.id}'
                self.curr_fea_graph.vertices[pos] = new_v # add node with coordinate and index info
            new_vertices.append(new_v) 


        # Check line overlap with existing edge using maximal edge representation 
        # check horizontal edges 
        h1 = (new_vertices[0], new_vertices[1]) # Vertex objects 
        h2 = (new_vertices[3], new_vertices[2])
        # check vertical edges
        v1 = (new_vertices[1], new_vertices[2])
        v2 = (new_vertices[0], new_vertices[3])
        # check diagonal lines
        d1, d2 = None, None
        if self.curr_frame_type == FrameType.DIAGONAL_LB_RT:
            d1 = (new_vertices[0], new_vertices[2])
        elif self.curr_frame_type == FrameType.DIAGONAL_LT_RB:
            d2 = (new_vertices[1], new_vertices[3])
        elif self.curr_frame_type == FrameType.DOUBLE_DIAGONAL:
            d1 = (new_vertices[0], new_vertices[2])
            d2 = (new_vertices[1], new_vertices[3])
        segments = {
                    'horizontal': [h1, h2],
                    'vertical': [v1, v2],
                    'LB_RT': [d1],
                    'LT_RB': [d2]
                }
        for direction, segs in segments.items():
            self.curr_fea_graph.combine_segments(segs, direction)

        # Update edge list with new line segments
        # get minimal edge list from each maximal edge
        maximal_edges = self.curr_fea_graph.maximal_edges
        all_edges = []
        for dir in maximal_edges:
            self.curr_fea_graph.merge_maximal_edges() # update maximal edges that are merged from new frame 
            for me in maximal_edges[dir]:
                # print(f'extracting edge from maximal edge : {me}')
                all_edges.extend(me.get_edge_list()) # get list of tuples of vertex indices
        self.curr_fea_graph.edges = all_edges


    # Check if in place change is equlivalent to this 
    # G = FEAGraph(vertices=copy.deepcopy(self.vertices), 
            #           supports = copy.deepcopy(self.supports),
            #           edges=copy.deepcopy(self.edges), 
            #           maximal_edges=copy.deepcopy(self.maximal_edges),
            #           loads = self.get_loads(), # update default and external(if connected) frame loads 
            #     )
            # self.curr_fea_graph = G

    def _update_load(self, new_vert):
        '''
        Input:
            new_vert : Vertex Object
        '''
        # TODO Update loads
        # loads = []
        # Iterate over all vertices in the graph
        # for vertex in self.vertices.values(): # Vertex objects
        # Start with the default load applied to the node
        load = [0.0, self.per_node_load, 0.0]

        # Check if this vertex's coordinates match any target load coordinates
        if new_vert.coordinates in self.target_loads:
            # If it's a target node, add the final load to the default load (element-wise)
            target_load = self.target_loads[new_vert.coordinates]
            load = [load[i] + target_load[i] for i in range(len(load))]

        self.curr_fea_graph.loads.append((new_vert.id, load))
        # Append the vertex id and load to the final loads list
        # loads.append((vertex.id, load))
