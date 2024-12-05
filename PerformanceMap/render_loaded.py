import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from TrussFrameMechanics.trussframe import FrameShapeType, FrameStructureType, TrussFrameRL
'''
Used to render from loaded hdf5 file
'''

class RenderLoaded:
    '''
    Inherits properties from env to render episode from loaded hdf5 file
    '''
    def __init__(self, render_properties) -> None:
        self.figsize = render_properties["figsize"] 
        # Create the figure and axes
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.board_size_x = render_properties["board_size_x"]
        self.board_size_y = render_properties["board_size_y"]
        self.allowable_deflection = render_properties["allowable_deflection"]
        self.frame_size = render_properties["frame_size"]

        self.curr_fea_graph = None
        self.frames = None
    
    def render_loaded(self, save_path, loaded_fea_graph, loaded_frames):
        '''
        given loaded_fea_graph, loaded_frames, loaded_frame_grid from load_episode_hdf5(hdf5_filename, eps_idx)
        render using derivative of render, render_frame, draw_fea_graph, draw_truss_analysis from cantileverenv_v0.py
        '''
        # set class values 
        self.curr_fea_graph = loaded_fea_graph
        self.frames = loaded_frames
        self.render_frame_loaded()
        plt.savefig(save_path, bbox_inches='tight')
        # plt.show()

    def render_frame_loaded(self):
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

        # # Highlight valid position cells
        # for frame_x, frame_y in self.valid_pos:
        #     x , y = self.framegrid_to_board(frame_x, frame_y)
        #     rect = patches.Rectangle((x - self.frame_size//2, y - self.frame_size//2), 
        #                                 self.frame_size, self.frame_size, 
        #                                 linewidth=0, edgecolor='lightblue', facecolor='lightblue')
        #     self.ax.add_patch(rect)
        
        # Draw current fea graph
        self.draw_fea_graph_loaded() # update self.fig, self.ax with current fea graph 

        # if len(self.curr_fea_graph.displacement) != 0 : # check if displacement has been analyzed 
        self.draw_truss_analysis_loaded() # last plot has displaced structure 
        self.ax.text(
                        0.5, -0.05,  # x=0.5 centers the text, y=0.01 places it at the bottom
                        f'Allowable Deflection : {self.allowable_deflection:.4f} m',
                        color='black',
                        fontsize=12,
                        ha='center',  # Center-aligns the text horizontally
                        transform=self.ax.transAxes  # Use axis coordinates
                    )

    def draw_fea_graph_loaded(self):
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
                self.ax.add_patch(patches.Rectangle((trussframe.x - self.frame_size//2, trussframe.y - self.frame_size//2), self.frame_size, self.frame_size, color='black', lw=5, fill=False))
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

        # Draw external forces as red arrows
        for coord, load in self.curr_fea_graph.external_loads.items():
            force_magnitude = (load[0]**2 + load[1]**2 + load[2]**2)**0.5
            if force_magnitude > 0:
                arrow_dx = load[0] * 0.01
                arrow_dy = load[1] * 0.01
                arrow_tail_x = coord[0] - arrow_dx
                arrow_tail_y = coord[1] - arrow_dy

                self.ax.arrow(arrow_tail_x, arrow_tail_y, arrow_dx, arrow_dy+0.1, head_width=0.2, head_length=0.2, fc='black', ec='black', linewidth=1.5)
                self.ax.text(arrow_tail_x, arrow_tail_y + 0.1, f"{force_magnitude:.2f} kN", color='black', fontsize=12)

    def draw_truss_analysis_loaded(self):
        '''
        Used within take step after episode as ended with connection
        Given that displacement has been updated
        Overlay displaced truss to plot by updating self.fig and self.ax based on self.curr_fea_graph.displacement
        Overlay failed elements in red based on self.curr_fea_graph.failed_elements
        '''
        displacement_scale = 10 # scale displacement for visualization 
        
        # Get Displaced vertices
        displaced_vertices = {} # node id : (x, y)
        max_disp = None # (node_id, displacement magnitude) 
        for i, (coord, V) in enumerate(self.curr_fea_graph.vertices.items()):
            dx, dy = self.curr_fea_graph.displacement[i][:2] * displacement_scale # Scale down if necessary for visualization
            # Calculate the displacement magnitude
            d_mag = np.sqrt((dx/displacement_scale)**2 + (dy/displacement_scale)**2)
            if max_disp == None or d_mag >= max_disp[1]:
                max_disp = (V.id, d_mag) 
            # print(f'displacement for node {i} is {dx}, {dy}')
            new_x = coord[0] + dx
            new_y = coord[1] + dy

            displaced_vertices[V.id] = (new_x, new_y)
            self.ax.add_patch(patches.Circle((new_x, new_y), radius=0.05, color='blue', alpha=0.8))
            # Add text showing displacement magnitude next to each circle
            self.ax.text(new_x + 0.1, new_y + 0.1, f'{d_mag:.3f}', color='gray', fontsize=8)
        
        # Connect deflected nodes with edges
        for edge in self.curr_fea_graph.edges:
            start_id, end_id = edge  # node ids
            start_coord = displaced_vertices[start_id]
            end_coord = displaced_vertices[end_id]

            # Plot the deflected truss member
            self.ax.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]],
                    color='blue', linestyle='--', linewidth=1)
        for edge in self.curr_fea_graph.failed_elements:
            start_id, end_id = edge
            start_coord = displaced_vertices[start_id]
            end_coord = displaced_vertices[end_id]
            self.ax.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]],
                    color='red', linestyle='-', linewidth=3)
        
        # Highlight max displacement by adding colored text
        maxd_x, maxd_y = displaced_vertices[max_disp[0]]
        maxd_value = max_disp[1]
        if maxd_value >= self.allowable_deflection:
            self.ax.text(maxd_x+0.1, maxd_y+0.2, f'{maxd_value:.3f}', color='red', fontsize=11)
        else:
            self.ax.text(maxd_x+0.1, maxd_y+0.2, f'{maxd_value:.3f}', color='green', fontsize=11)
