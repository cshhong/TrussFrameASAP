'''
TrussFrames of different types can be drawn on a grid with overlap.

TODO make merges efficient
TODO make memory (storing G sequences) efficient

DONE link julia with python 
Pycall : have to compile python with shared library which is difficult within conda environment (https://www.juliabloggers.com/how-to-call-julia-code-from-python/)
JuliaCall / PythonCall : https://juliapy.github.io/PythonCall.jl/stable/juliacall/


Graph Nodes
G.add_node(vertex_name, type='vertex', pos=vertex, square_ind=idx, idx=id_vertex)
G.add_node(edge_node, type='edge', pos=(G.nodes[start]['pos'],G.nodes[end]['pos']), square_idx=idx, idx=id_edge)
G.add_node(intersection_node, type='vertex', intersection=set([edge1_key, edge2_key]), pos=intersection)

'''

import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import networkx as nx
# from networkx.algorithms import isomorphism
import numpy as np
import math 

from vertex import Vertex
from maximaledge import MaximalEdge
from feagraph import FEAGraph
from trussframe import FrameType, TrussFrame


import random
from enum import Enum, auto
import copy

import juliacall
# from pythonAsap import solve_truss_from_graph
import pythonAsap 

import ipywidgets as widgets
from IPython.display import display

import generate_bc

# class FrameType(Enum):
#     '''
#     # Checking the values assigned by auto()
#     print(FrameType.SQUARE)            # Output: FrameType.SQUARE
#     print(FrameType.SQUARE.value)      # Output: 1
#     print(FrameType.DIAGONAL_LT_RB)    # Output: FrameType.DIAGONAL_LT_RB
#     print(FrameType.DIAGONAL_LT_RB.value)  # Output: 2
#     '''
#     SQUARE = auto() # assigns an index as value automatically
#     DIAGONAL_LT_RB = auto()
#     DIAGONAL_LB_RT = auto()
#     DOUBLE_DIAGONAL = auto()
#     def __str__(self):
#         return self.name
    
# DEFAULT_FRAME_TYPE = FrameType.DIAGONAL_LT_RB

# class TrussFrame:
#     '''
#     TrussFrame object with centroid position and frame type
#     By Default is set to type with FrameType.DIAGONAL_LT_RB(diagonal brace from left top to right bottom)
    
#     '''
#     def __init__(self, pos, frame_type=DEFAULT_FRAME_TYPE):
#         self.x = pos[0] # local (x,y) center grid position of the frame
#         self.y = pos[1] # local (x,y) center grid position of the frame
#         self.frame_type = frame_type  # Type of the frame

class TrussFrameApp:
    '''
    self.graphs = [] # sequence of graph structures that is created at each addition of frame
    self.vertices = dict() # dictionary of vertices with coordinate as key and Vertex object as value
    self.edges = [] # adjacency list of tuples of vertex indices pairs
    self.maximal_edges = {
        'horizontal': [],
        'vertical': [],
        'LB_RT': [],
        'LT_RB': []
    }
    - Vertex:
        Represents a node in the graph, uniquely identified by an ID and
        defined by its coordinates in a 2D space. Each vertex stores the edges
        connected to it. Vertices are ordered based on their coordinates.
    
    - MaximalEdge:
        Represents a sequence of vertices connected in a specific direction
        ('horizontal', 'vertical', 'LB_RT', or 'LT_RB'). A maximal edge can 
        consist of multiple smaller edges. It stores a list of vertices that 
        define the edge's endpoints and manages these as truss frames are added.
    
    - Edge List:
        Stores individual edges (line segments) extracted from the maximal edges.
        Each edge is a tuple of vertex indices representing the endpoints of 
        that edge. Edges are dynamically generated by extracting them from the 
        maximal edges after the graph is updated.
    
    - FEAGraph (G):
        A dictionary that stores the entire graph at various stages. It consists
        of the following key components:
        vertices : A dictionary where keys are coordinates and values are Vertex objects (read-only).
        supports : list of vertex idx where the nodes in the frame are supports / pinned (as opposed to free)
        edges : An adjacency list of tuples representing edges, where each tuple contains
            vertex indices (read-only).
        maximal_edges : A dictionary of maximal edges grouped by direction. Each direction contains
            a list of MaximalEdge objects (read-only).
        The graph is updated as truss frames are added, and deepcopies of the
        vertices, edges, and maximal edges are stored over time.
    '''
    def __init__(self, board_size_x, board_size_y, cell_size, square_size, figsize, verbose=True, block_type_mode=False, seed=None):
        # settings for game 
        self.block_type_mode = block_type_mode
        self.board_size_x = board_size_x
        self.board_size_y = board_size_y
        self.cell_size = cell_size
        self.square_size = square_size

        # self.squares = []
        self.frames = [] # list of TrussFrame objects 
        self.verbose = verbose
        
        # Drawing board object and events
        self.fig = None
        self.figsize = figsize
        self.ax = None
        self.click_event_id = None
        self.key_event_id = None

        # TrussFrame type
        DEFAULT_FRAME_TYPE = FrameType.DIAGONAL_LT_RB
        self.curr_frame_type = DEFAULT_FRAME_TYPE # current frame type to be drawn selected by user keypress event

        # Done button in drawing board
        self.done_button_ax = None
        self.done_button = None
        
        # series of graphs 
        self.graphs = [] # sequence of FEAGraph objects that is created at each addition of frame
        self.vertices = dict() # dictionary of vertices with coordinate as key and Vertex object as value
        # self.supports = [1,2] # list of vertex.id that are designated as supports / pinned as opposed to free
        # self.supports = {} # dictionary of vertex.id : grid coordinates that are designated as supports / pinned as opposed to free
        self.edges = [] # adjacency list of tuples of vertex indices pairs
        self.maximal_edges = {
            'horizontal': [],
            'vertical': [],
            'LB_RT': [], 
            'LT_RB': []
        }

        # Set Cantilever Env
        self.default_load = -0.4 # N applied per node
        default_frames, supports, target_load = generate_bc.set_cantilever_env(self.board_size_x, self.square_size, seed=seed)
        # print(f'default_frames : {default_frames}, supports : {supports}, target load : {target_load}')
        self.target_loads = target_load # Dictionary where key is the (x, y) grid coordinate and value is the [x, y, z] force magnitude
        self.supports = supports # dictionary of vertex.id : grid coordinates that are designated as supports / pinned as opposed to free
        # add the inital frame(s) generated by env_cantilever
        for f_coords in default_frames:
            new_frame = TrussFrame(f_coords)
            self.frames.append(new_frame)
            self.update_curr_state(new_frame)
            all_loads = self.get_loads()
            G = FEAGraph(vertices=copy.deepcopy(self.vertices), 
                      supports = copy.deepcopy(self.supports),
                      edges=copy.deepcopy(self.edges), 
                      maximal_edges=copy.deepcopy(self.maximal_edges),
                      loads = all_loads,
                )
            self.graphs.append(G)

    def view_drawingboard(self):
        '''
        Create interactive drawing board where user can click to add squares
        '''
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_xlim([0, self.board_size_x])
        self.ax.set_ylim([0, self.board_size_y])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xticks(range(self.board_size_x + 1))
        self.ax.set_yticks(range(self.board_size_y + 1))

         # Draw grid lines
        self.ax.grid(True, which='both', color='lightblue', linestyle='-', linewidth=1)
           # Draw green lines at even intervals
        for i in range(0, self.board_size_x + 1, 2):
            self.ax.axvline(x=i, color='lightblue', linestyle='-', linewidth=2)
        for j in range(0, self.board_size_y + 1, 2):
            self.ax.axhline(y=j, color='lightblue', linestyle='-', linewidth=2)

        # Draw boundary conditions from env that populated App values
        self.update_drawing() 
        
        # Connect the button press event (add frame)
        self.click_event_id = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Connect the keypress event (select frame type)
        self.key_event_id = self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
        # Add text element to display current frame type
        frame_type_text = f'{self.curr_frame_type.name}' if self.curr_frame_type else 'PRESS NUM KEY TO SELECT'
        self.frame_type_text = self.ax.text(1.05, 1.0, f'TrussFrame Type: \n{frame_type_text}',
                                    transform=self.ax.transAxes, fontsize=12, verticalalignment='top')

        # Add "Done" button
        self.done_button_ax = plt.axes([0.90, 0.05, 0.1, 0.05])
        self.done_button = plt.Button(self.done_button_ax, 'Done')
        self.done_button.on_clicked(self.on_done)
        
        # Create a circle that will follow the cursor
        self.cursor_circle = patches.Circle((0, 0), radius=self.cell_size / 7, color='blue', fill=False, linewidth=2)
        self.ax.add_patch(self.cursor_circle)
        # Connect the motion_notify_event to update the circle's position
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        self._draw_force_arrows(self.ax) # Draw Target Load arrows

        plt.grid(True)
        plt.show()

    def on_mouse_move(self, event):
        '''
        Update the circle's position to follow the cursor.
        '''
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            # Update the circle's center to the current mouse position
            # (allow half overlapping frames)
            # self.cursor_x = round(event.xdata / self.cell_size) * self.cell_size
            # self.cursor_y = round(event.ydata / self.cell_size) * self.cell_size
            # (frames are strictly adjacent)
            self.cursor_x = round((event.xdata + self.square_size //2) / self.square_size) * self.square_size - self.square_size //2
            self.cursor_y = round((event.ydata + self.square_size //2)/ self.square_size) * self.square_size - self.square_size //2


            # snap to board bounds
            self.cursor_x = min(max(self.cursor_x, self.square_size//2), self.board_size_x - self.square_size//2)
            self.cursor_y = min(max(self.cursor_y, self.square_size//2), self.board_size_y - self.square_size//2)

            self.cursor_circle.center = (self.cursor_x, self.cursor_y)
            self.fig.canvas.draw_idle()

    def on_click(self, event):
        '''
        Add frame with the click event on the grid at snapped cursor location. 
        cursor location is the centroid of the truss frame.
        update graph with new frame
        '''
        # Check if click is within the grid bounds and not on the button
        if event.inaxes != self.done_button_ax and event.xdata is not None and event.ydata is not None:
            # Save all frame objects
            new_frame = TrussFrame((self.cursor_x, self.cursor_y), self.curr_frame_type)
            self.frames.append(new_frame)

            # # create new graph
            self.update_curr_state(new_frame)

            all_loads = self.get_loads()
            # print(all_loads)

            G = FEAGraph(vertices=copy.deepcopy(self.vertices), 
                      supports = copy.deepcopy(self.supports),
                      edges=copy.deepcopy(self.edges), 
                      maximal_edges=copy.deepcopy(self.maximal_edges),
                      loads = all_loads,
                )

            self.graphs.append(G)

            # Draw frame objects on the grid
            self.update_drawing() 

    def on_keypress(self, event):
        '''
        Change current frame type based on key press.
        '''
        if self.block_type_mode:
            # print(f'key pressed : {event.key}')
            if event.key == '1':
                self.curr_frame_type = FrameType.SQUARE
            elif event.key == '2':
                self.curr_frame_type = FrameType.DIAGONAL_LT_RB
            elif event.key == '3':
                self.curr_frame_type = FrameType.DIAGONAL_LB_RT
            elif event.key == '4':
                self.curr_frame_type = FrameType.DOUBLE_DIAGONAL
            else:
                self.curr_frame_type = FrameType.SQUARE

            # Update the existing text element
            self.frame_type_text.set_text(f'TrussFrame Type: \n{self.curr_frame_type.name}')
        else:
            self.curr_frame_type = FrameType.DIAGONAL_LT_RB

        self.fig.canvas.draw()


    def update_curr_state(self, new_frame):
        '''
        Input 
            new_frame : TrussFrame object (centroid, frame type)
        
        Update current graph with added truss frame so that existing node indices are preserved
        1. merge overlapping new nodes with existing nodes
        2. check line overlap with existing edge using maximal edge representation 
        3. update edge list with new line segments

        update self.vertices
        update self.edge_list
        '''
        half_size = self.square_size / 2
    
        # Calculate the positions of the four vertices
        vert_pos = [
            (new_frame.x - half_size, new_frame.y - half_size),  # Bottom-left
            (new_frame.x + half_size, new_frame.y - half_size),  # Bottom-right
            (new_frame.x + half_size, new_frame.y + half_size),  # Top-right
            (new_frame.x - half_size, new_frame.y + half_size)   # Top-left
        ]
        
        # 1. Merge overlapping new nodes with existing nodes
        new_vertices = [] # node ind in order of bottom-left, bottom-right, top-right, top-left
        vertices = self.vertices
        for pos in vert_pos:
            # get node index if node exists in self.nodes
            if pos in vertices:
                vert = vertices[pos]
            else: # If the node does not exist in the current graph, create new node
                vert_idx = len(vertices)  # Assign a new index to the node
                vert = Vertex(pos)
                assert len(vertices)+1 == vert.id, f'node index mismatch between {len(self.vertices)} and {vert.id}'
                self.vertices[pos] = vert # add node with coordinate and index info
            new_vertices.append(vert) 
        # for pos in vert_pos:
        #     # get node index if node exists in self.nodes
        #     if pos in self.vertices:
        #         vert = self.vertices[pos]
        #     else: # If the node does not exist in the current graph, create new node
        #         vert_idx = len(self.vertices)  # Assign a new index to the node
        #         vert = Vertex(pos)
        #         assert len(self.vertices)+1 == vert.id, f'node index mismatch between {len(self.vertices)} and {vert.id}'
        #         self.vertices[pos] = vert # add node with coordinate and index info
        #     new_vertices.append(vert)  

        # 2. Check line overlap with existing edge using maximal edge representation 
        # check horizontal edges 
        h1 = (new_vertices[0], new_vertices[1]) # Vertex objects 
        h2 = (new_vertices[3], new_vertices[2])
        # self._combine_segments([h1, h2], 'horizontal')
        # check vertical edges
        v1 = (new_vertices[1], new_vertices[2])
        v2 = (new_vertices[0], new_vertices[3])
        # self._combine_segments([v1, v2], 'vertical')
        # check diagonal lines
        d1, d2 = None, None
        if self.curr_frame_type == FrameType.DIAGONAL_LB_RT:
            d1 = (new_vertices[0], new_vertices[2])
        elif self.curr_frame_type == FrameType.DIAGONAL_LT_RB:
            d2 = (new_vertices[1], new_vertices[3])
        elif self.curr_frame_type == FrameType.DOUBLE_DIAGONAL:
            d1 = (new_vertices[0], new_vertices[2])
            d2 = (new_vertices[1], new_vertices[3])
        # self._combine_segments([d1], 'LB_RT')
        # self._combine_segments([d2], 'LT_RB')
        segments = {
                    'horizontal': [h1, h2],
                    'vertical': [v1, v2],
                    'LB_RT': [d1],
                    'LT_RB': [d2]
                }
        for direction, segs in segments.items():
            self._combine_segments(segs, direction)
                        
        # 3. Update edge list with new line segments
        # get minimal edge list from each maximal edge
        all_edges = []
        for dir in self.maximal_edges:
            self._merge_maximal_edges() # update maximal edges that are merged from new frame 
            for me in self.maximal_edges[dir]:
                # print(f'extracting edge from maximal edge : {me}')
                all_edges.extend(me.get_edge_list()) # get list of tuples of vertex indices
        self.edges = all_edges

        # print(f'edges : {self.edges}')
        # self._print_maximal_edges()

    def get_loads(self):
        '''
        Used within on_click
        At addition of new frame after updating vertices and edges
            - at all nodes apply default load 
            - apply final loads if reached
                check if target node position is occupied (means that the structure is complete?)
                and apply final loads 
        
        return nodes in the format of list of tuples
        loads = [
        (4, [-4000.0, -8000.0, 0.0]),  # Load applied to node 3 (with index 2)
        (3, [-4000.0, -8000.0, 0.0]),  # Load applied to node 4 (with index 3)
        ]
        '''
        loads = []
        
        # Iterate over all vertices in the graph
        for vertex in self.vertices.values():
            # Start with the default load applied to the node
            load = [0.0, self.default_load, 0.0]

            # Check if this vertex's coordinates match any target load coordinates
            if vertex.coordinates in self.target_loads:
                # If it's a target node, add the final load to the default load (element-wise)
                target_load = self.target_loads[vertex.coordinates]
                load = [load[i] + target_load[i] for i in range(len(load))]

            # Append the vertex id and load to the final loads list
            loads.append((vertex.id, load))
        
        return loads

    def _merge_maximal_edges(self):
        '''
        After adding new frame which is merged to first overlapping maximal edge
        Merge maximal edges that are connected from new frame 
        '''
        for dir in self.maximal_edges:
            # check for pairs of maximal edges in the same direction
            for i in range(len(self.maximal_edges[dir])-2):
                for j in range(i + 1, len(self.maximal_edges[dir])-1):
                    me1 = self.maximal_edges[dir][i]
                    me2 = self.maximal_edges[dir][j]
                    if me1.is_connected(me2):
                        # Merge the two maximal edges
                        me1.vertices += me2.vertices 
                        me1.vertices = sorted(set(me1.vertices))
                        self.maximal_edges[dir].remove(me2)
                
    def _combine_segments(self, segments, direction):
        '''
        Combine segments with existing maximal edges in the specified direction.
        Updates self.maximal_edges[direction] in place.
        '''
        # Check if the list of maximal edges is not empty
        if len(self.maximal_edges[direction]) != 0:
            for seg in segments:
                if seg == None:
                    continue # skip if segment is None
                merged = False  # Track if merging occurs for this pair
                # Try to merge with existing maximal edges
                for me in self.maximal_edges[direction]:
                    if me.merge_segment(*seg):  # Try to merge
                        merged = True
                        break  # Stop after the first successful merge
                # If not merged, create a new maximal edge
                if not merged:
                    new_me = MaximalEdge(direction)  # Create new maximal edge object
                    new_me.vertices += list(seg)
                    self.maximal_edges[direction].append(new_me)

        else: # If the list is empty, create new maximal edges for all segments
            for seg in segments:
                if seg == None:
                    continue # skip if segment is None
                new_me = MaximalEdge(direction)  # Create new maximal edge object
                new_me.vertices += list(seg)
                self.maximal_edges[direction].append(new_me)

    def _print_maximal_edges(self):
        '''
        Print maximal edges of all directions with a list of vertices by index.
        '''
        for direction, edges in self.maximal_edges.items():
            print(f"Maximal Edges (Direction: {direction}):")
            for edge in edges:
                vertex_indices = [vertex.id for vertex in edge.vertices]
                print(f"  Edge with Vertices {vertex_indices}")
            if not edges:
                print("  No edges in this category.")
            print()  # Blank line for better readability

    def update_drawing(self):
        '''
        Based on updated graph 
        self.vertices, self.edges, self.maximal_edges
        draw the updated graph on the grid
        '''
        text_offset = 0.1
        # Draw vertices
        for coord, vertex in self.vertices.items():
            if vertex.id in list(self.supports.keys()):
                self.ax.add_patch(patches.Rectangle((coord[0] - 0.1, coord[1] - 0.1), 0.2, 0.2, color='blue', lw=1.5, fill=True))
            else:
                self.ax.add_patch(patches.Circle(coord, radius=0.1, color='blue', lw=1.5, fill=False ))
            self.ax.text(coord[0]-text_offset, coord[1]+text_offset, 
                         str(vertex.id), 
                         fontsize=10, ha='right', color='black')

        # Draw maximal edges (optional, if visually distinct from normal edges)
        for direction, edges in self.maximal_edges.items():
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

    def on_done(self, event):
        plt.close(self.fig)

    def view_graphs(self):
        '''
        executed after clicking done button from app.view_drawingboard
        Display the series of truss frame graphs on the same plot
        '''

        self.fig.canvas.mpl_disconnect(self.click_event_id)  # Disconnect the click event
        self.fig.canvas.mpl_disconnect(self.key_event_id)  # Disconnect the click event

        num_subplots = len(self.graphs)
        ncols = min(num_subplots, 4)
        nrows = (num_subplots + ncols - 1) // ncols

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=self.figsize)
        # Adjust the spacing between subplots
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.01, hspace=0.01)
        
        if isinstance(axs, np.ndarray):
            axs = axs.flatten()

        for i, G in enumerate(self.graphs):
            # TODO handle single graph
            # if isinstance(axs, list):
            #     ax = axs[i]
            self._draw_graph(i, ax=axs[i])

        # Remove unused axes
        for ax in axs[num_subplots:]:
            ax.axis('off')
        plt.show()

    def _draw_graph(self, graph_idx, ax, simplified=False):
        '''
        Plot truss frame graph without grid on ax object
        Input : 
            graph_idx : index in self.graphs
        '''
        ax.set_xlim([0, self.board_size_x])
        ax.set_ylim([0, self.board_size_y])
        ax.set_aspect('equal', adjustable='box')
        # Turn off the ticks and tick labels
        ax.set_xticks([])  # Turn off x-axis ticks
        ax.set_yticks([])  # Turn off y-axis ticks

        # Draw frames as squares
        for frame in self.frames[:graph_idx+1]: # draw up to current graph idx
            # The lower-left corner of the square
            lower_left = (frame.x - 1, frame.y - 1)
            # Add a square patch of size 2x2
            ax.add_patch(patches.Rectangle(lower_left, 2, 2, fill=True, edgecolor='blue', facecolor='lightblue'))
            
            # Add the frame index in the center of the square
            ax.text(frame.x, frame.y, str(self.frames.index(frame)), color='white', fontsize=12, ha='center', va='center')

        text_offset = 0.1
        # Draw vertices
        G = self.graphs[graph_idx]
        for coord, vertex in G.vertices.items():
            if vertex.id in list(self.supports.keys()):
                ax.add_patch(patches.Rectangle((coord[0] - 0.2, coord[1] - 0.2), 0.4, 0.4, color='black', lw=0.01, fill=True))
            else:
                ax.add_patch(patches.Circle(coord, radius=0.1, color='black', lw=0.01, fill=True))
            if not simplified:
                ax.text(coord[0]-text_offset, coord[1]+text_offset, 
                            str(vertex.id), 
                            fontsize=10, ha='right', color='black')
                


        # Draw maximal edges (optional, if visually distinct from normal edges)
        # for direction, edges in G['maximal_edges'].items():
        for direction, edges in G.maximal_edges.items():
            for edge in edges:
                if len(edge.vertices) >= 2:
                    # Get start and end vertices from the list of vertices
                    start_me = edge.vertices[0]
                    end_me = edge.vertices[-1]
                    
                    # Draw the line connecting the start and end vertices
                    ax.plot([start_me.coordinates[0], end_me.coordinates[0]], 
                                [start_me.coordinates[1], end_me.coordinates[1]], 
                                color='black', linestyle='-', linewidth=1)
                else:
                    print(f"Warning: Maximal edge in direction {direction} has less than 2 vertices and cannot be drawn.")


    def view_truss_analysis(self, displacements):
        '''
        Given Julia truss analysis for a series of graphs, 
        plot truss analysis for each graph in the series with fixed subplot size and margin along x-axis.
        '''

        num_subplots = len(self.graphs)
        n_cols = 4
        n_rows = (num_subplots + n_cols - 1) // n_cols  # Compute the required number of rows

        # Fixed subplot size and margins
        subplot_width = 5  # Fixed width of each subplot
        subplot_height = 4  # Fixed height of each subplot
        x_margin = 0.1  # Fixed margin between subplots along x-axis

        # Calculate figure size based on fixed subplot size and the number of columns/rows
        fig_width = subplot_width * n_cols + (n_cols - 1) * x_margin
        fig_height = subplot_height * n_rows

        # Create a scrolling widget using ipywidgets
        output_widget = widgets.Output()

        with output_widget:
            # Create the figure with adjusted dimensions
            self.fig, self.ax = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
            # self.fig, self.ax = plt.subplots(n_rows, n_cols, figsize=self.figsize)
            
            # Flatten the 2D array of axes for easier indexing
            ax_list = self.ax.ravel()

            # Display each graph with its corresponding displacement
            for i, G in enumerate(self.graphs):
                disp = displacements[i]
                ax = ax_list[i]
                self._draw_truss_analysis(ax, i, disp, simplified=True)

            # Hide unused subplots
            for j in range(num_subplots, len(ax_list)):
                self.fig.delaxes(ax_list[j])

            # Adjust layout: `wspace` controls the space between columns (horizontal margin)
            plt.subplots_adjust(wspace=x_margin / subplot_width, hspace=0.1)  # Keep vertical gaps (hspace) at 0.5
            plt.tight_layout()

            plt.show()

        # Create a scrollable container
        scrollable_box = widgets.Box([output_widget], layout=widgets.Layout(overflow_y='scroll', width = '200px', height='200px'))
        
        # Display the scrollable box
        display(scrollable_box)


    # Overlay with deflected truss - preserve truss element length to create curves?
    def _draw_truss_analysis(self, ax, graph_idx, displacement, simplified=False):
        '''
        Given displacement for a graph, plot graph, nodal displacement, and max displacement in the provided ax (subplot).
        
        # G : FEAGraph object
        graph_idx : index within self.graphs
        displacement : list of nodal displacement for each node in node index order.
                    Each entry can be either a scalar (magnitude) or a vector (e.g., [dx, dy])
                    corresponding to the displacement at the node.
        '''

        displacement_scale = 100 # scale displacement for visualization 
        force_arrow_scale = 0.1 # Adjust the scale factor as necessary for kN visualization
        
        # Set up the axis for the current subplot
        ax.set_xlim([0, self.board_size_x])
        ax.set_ylim([0, self.board_size_y])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])  # Turn off x-axis ticks
        ax.set_yticks([])  # Turn off y-axis ticks

        G = self.graphs[graph_idx]
        # Draw the original truss graph 
        self._draw_graph(graph_idx, ax=ax, simplified=simplified)

        # Find the maximum displacement index and value
        max_disp_index = np.argmax([np.linalg.norm(d[:2]) for d in displacement])
        max_disp_value = np.linalg.norm(displacement[max_disp_index][:2])

        # Update vertex positions based on displacement and create a new mapping of vertices
        displaced_vertices = {}
        for i, (coord, vertex) in enumerate(G.vertices.items()):
            dx, dy = displacement[i][:2] * displacement_scale # Scale down if necessary for visualization
            # print(f'displacement for node {i} is {dx}, {dy}')
            new_x = coord[0] + dx
            new_y = coord[1] + dy

            displaced_vertices[vertex.id] = (new_x, new_y)
            ax.add_patch(patches.Circle((new_x, new_y), radius=0.05, color='blue', alpha=0.8))
            # Calculate the displacement magnitude
            displacement_magnitude = np.sqrt(dx**2 + dy**2)
            # Add text showing displacement magnitude next to each circle
            ax.text(new_x + 0.1, new_y + 0.1, f'{displacement_magnitude:.2f}', color='gray', fontsize=8)

        # Draw deflected truss using G.edges
        for edge in G.edges:
            start_vertex, end_vertex = edge  # Assuming edges are defined as tuples of vertex ids
            start_coord = displaced_vertices[start_vertex]
            end_coord = displaced_vertices[end_vertex]

            # Plot the deflected truss member
            ax.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]],
                    color='blue', linestyle='--', linewidth=1)
                
        # Draw nodal loads as red arrows in kN
        self._draw_force_arrows(ax)
            
    def _draw_force_arrows(self, ax, force_arrow_scale=0.025):
        """
        Draw force arrows on the plot based on the self.target_loads at node coordinates.

        Parameters:
        - ax: matplotlib axis to draw the arrows on.
        - target_loads: dictionary with node coordinates as keys and force vectors as values.
        - force_arrow_scale: scaling factor for arrow length for better visualization.
        """
        for node_coord, force_vector in self.target_loads.items():
            fx, fy, _ = force_vector  # Extract x and y components (ignore z)

            # Forces in kN (assuming input is already in kN)
            fx_kN = fx 
            fy_kN = fy 

            # Scale down the force vector for better visibility in the plot
            arrow_dx = fx_kN * force_arrow_scale
            arrow_dy = fy_kN * force_arrow_scale

            # Calculate arrow tail position (so the arrow tip is at the node coordinate)
            arrow_tail_x = node_coord[0] - arrow_dx 
            arrow_tail_y = node_coord[1] - arrow_dy + 0.3
            
            # Draw the arrow representing the force
            ax.arrow(arrow_tail_x, arrow_tail_y, arrow_dx, arrow_dy, 
                    head_width=0.2, head_length=0.1, fc='red', ec='red', width=0.05
                    )

            # Calculate the magnitude of the force
            force_magnitude_kN = np.sqrt(fx_kN**2 + fy_kN**2)

            # Display the force magnitude near the arrow
            # ax.text(node_coord[0] + arrow_dx + 0.2, node_coord[1] + arrow_dy + 0.2, f"{force_magnitude_kN:.2f} kN",
            #         color='red', fontsize=6, fontweight='bold')
            ax.text(arrow_tail_x, arrow_tail_y + 0.1, f"{force_magnitude_kN:.2f} kN",
                    color='red', fontsize=8, fontweight='bold')
            
def main():

    board_size_x = 20 
    board_size_y = 10
    cell_size = 1
    square_size = 2
    figsize = (15, 8)
    verbose = True

    app = TrussFrameApp(board_size_x, board_size_y, cell_size, square_size, figsize, verbose, seed=39)
    app.view_drawingboard()
    # app.view_graphs()
    # print("FEAGraph created")
    # app.view_subgraph()

    # Solve truss model with ASAP
    jl = juliacall.newmodule("TrussFrameRL") 
    curr_env = jl.seval('Base.active_project()')
    print(f"The current active Julia environment is located at: {curr_env}")

    # Step 0: Initialize Julia session and import necessary Julia modules
    jl.seval('using AsapToolkit')
    jl.seval('using Asap')

    jl.include("truss_analysis.jl")
    jl.seval('using .TrussAnalysis')

    displacements = []

    for i, G in enumerate(app.graphs):
        disp = pythonAsap.solve_truss_from_graph(jl, G, G.loads) # return nodal displacement
        # print(f"displacement : {disp}")
        displacements.append(disp)

    # display displacement at each node 
    
    app.view_truss_analysis(displacements)



if __name__ == "__main__":
    main()