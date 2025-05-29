"""
Custom graph data structure that stores additional information needed for structural analysis
"""
from .maximaledge import MaximalEdge  # Relative import
from .trussframe import FrameShapeType, FrameStructureType     # Relative import
import numpy as np
    
class FEAGraph:
    """
    A data structure representing an immutable graph that consists of vertices,
    edges, and maximal edges. 
    Used for FEA (using ASAP)
        model = jl.create_and_solve_model_julia(node_coords, element_conns, support_idx, loads)
        node_coords : node coordinates in the order of ids
        element_conns : adjacency list with node ids
        support_idx : list of supports with node ids
        loads (including external) : (node.id, [load.x, load.y, load.z]) {Tuple{Int, Vector{Float64}}}

    Visualize structural model on board
        Node shape based on coordinate, is_free info
        Edge : connection represented as line between Node objects 

        
    
    Attributes:
    -----------
        
    vertices : A dictionary where keys are coordinates and values are Vertex objects.
        Vertex objects have attribute
            coordinates: A tuple representing the position of the vertex (e.g., (x, y) for 2D space).
            id: A unique integer identifier for each vertex, automatically assigned using a class-level counter (_id_counter).
            is_free : Boolean that represent whether the structural node is free or pinned
            load : load value on vertex
    supports : list of board coordinates of the nodes in the frame are supports / pinned (as opposed to free)
    (outdated) edges : An adjacency list of tuples representing edges, where each tuple contains vertex indices. 
    edges_dict : dictionary where keys : (v_1, v_2) Vertex objects, values : (outer diameter, inner wall thickness ratio)
    external_loads : dictionary where key : coordinate in board, value : load magnitude [load.x, load.y, load.z]
    displacement :  2D list of nodal displacement [x,y,z] for each node in node index order. Only non-empty upon fea (eps end)
    failed_elements : list of tuple (node_idx1, node_idx2, compression-0/tension-1)

    # Apply variations of sections
    edge_type_dict : dictionary where key : edge type int (weakest -> strongest), value : (outer diameter, inner wall thickness ratio) 
    
    edges_dict : dictionary where keys : (v_1, v_2) Vertex objects, values : (outer diameter, inner wall thickness ratio)
    
    """
    
    def __init__(self, vertices=None, supports=None, edges=None, external_loads=None, displacement=None, failed_elements=None, edges_dict=None, utilization=None):
        """
        Initializes the graph with dictionaries of vertices, edges, and maximal edges.
        """

        # self.default_node_load = default_node_load # kN
        # Decided with epside env init
        self.supports = supports if supports is not None else []
        self.external_loads = external_loads if external_loads is not None else {}
        
        self.vertices = vertices if vertices is not None else {}
        self.edges = edges if edges is not None else []

        if displacement is not None:
            self.displacement = displacement
        else:
            self.displacement = []
        
        if failed_elements is not None:
            self.failed_elements = failed_elements
        else:
            self.failed_elements = []

        if edges_dict is not None:
            self.edges_dict = edges_dict
        else:
            self.edges_dict = dict() 

        if utilization is not None:
            self.utilization = utilization
        else:
            self.utilization = []

    def __repr__(self):
        """Nicely formatted representation of the graph."""
        supports_repr = "\n".join([f"  {s}" for s in self.supports])
        externalloads_repr = "\n".join([f"  {coord}: {load_val}" for coord, load_val in self.external_loads.items()])
        vertices_repr = "\n".join([f"  {coord}: {v}" for coord, v in self.vertices.items()])
        edges_repr = "\n".join([f"  {e}" for e in self.edges])
        # displacement_repr = "\n".join([f"  {node_idx}: {displacement}" for node_idx, displacement in enumerate(self.displacement)])
        displacement_repr = "\n".join(
                            f"  {node_idx}: [{', '.join(f'{x:.2f}' for x in disp)}]"
                            for node_idx, disp in enumerate(self.displacement)
                        )
        failed_elements_repr = "\n".join([f"  {e}" for e in self.failed_elements])
        edge_dict_repr = "\n".join([f"  {edge}: {section}" for edge, section in self.edges_dict.items()])
        utilization_repr = "\n".join([f"  {idx}: {util}" for idx, util in enumerate(self.utilization)])

        return (
            # f"Default Node load : ({self.default_node_load})\n"
            f"Supports ({len(self.supports)}):\n{supports_repr}\n"
            f"External Loads ({len(self.external_loads)}):\n{externalloads_repr}\n"
            f"Vertices ({len(self.vertices)}):\n{vertices_repr}\n"
            # f"Vertices ({len(self.vertices)})\n"
            # f"Edges ({len(self.edges)}):\n{edges_repr}\n"
            # f"Edges ({len(self.edges)})\n"
            # f"Maximal Edges ({len(self.maximal_edges)}):\n{maximal_edges_repr}\n"
            f"Displacement ({len(self.displacement)}):\n{displacement_repr}\n)"
            # f"Failed Elements ({len(self.failed_elements)}):\n{failed_elements_repr}\n)"
            # f"Edge Dictionary ({len(self.edges_dict)}):\n{edge_dict_repr}\n"
            # f"Edge Type Dictionary ({len(self.edge_type_dict)}):\n{edge_type_dict_repr}\n"
            # f"Utilization ({len(self.utilization)}):\n{utilization_repr}\n"

        )

    def get_all_node_ids(self):
        """Get all integer value node ids from FEAGraph object"""
        return [vertex.id for vertex in self.vertices.values()]
    
    def combine_and_merge_edges(self, new_vertices=None, frame_structure_type=None):
        '''
        (overlapping vertices are merged already)
        Input
            frame_type : FrameShapeType object to indicate which vertices should be connected
            # new_vertices : List of Vertex objects in order of bottom-left, bottom-right, top-right, top-left
            new_vertices : List of Vertex objects in order of bottom-left, bottom-right, top-left, top-right
        Check overlapping edge segments with self.curr_fea_graph.edges_dict
        Update self.curr_fea_graph.edges_dict, in place
        used within update_curr_fea_graph
        '''
        assert frame_structure_type is not None, "frame_structure_type is None, must be provided"
        frame_type_shape = frame_structure_type.shape_type if frame_structure_type else frame_type_shape
        
        # check horizontal edges 
        h1 = (new_vertices[0], new_vertices[1]) # Vertex objects 
        h2 = (new_vertices[2], new_vertices[3])
        # check vertical edges
        v1 = (new_vertices[1], new_vertices[3])
        v2 = (new_vertices[0], new_vertices[2])
        # check diagonal lines
        d1, d2 = None, None
        if frame_type_shape == FrameShapeType.DIAGONAL_LB_RT:
            d1 = (new_vertices[0], new_vertices[3])
        elif frame_type_shape == FrameShapeType.DIAGONAL_LT_RB:
            d2 = (new_vertices[1], new_vertices[2])
        elif frame_type_shape == FrameShapeType.DOUBLE_DIAGONAL: # we only use this type for now
            d1 = (new_vertices[0], new_vertices[3])
            d2 = (new_vertices[1], new_vertices[2])
        segments = {
                    'horizontal': [h1, h2],
                    'vertical': [v1, v2],
                    'LB_RT': [d1],
                    'LT_RB': [d2]
                }
        
        # update edges without maximal edge, edge with section geometry
        for dir, segs in segments.items():
            for e in segs:
                if e != None:
                    self._update_edges_dict(e, dir, frame_structure_type)

    
    def _update_edges_dict(self, edge, direction, frame_structure_type):
        '''
        Update self.edges_dict with new segments in the specified direction.
        self.edges_dict : dictionary where keys : (v_1, v_2) Vertex objects, values : (outer diameter, inner wall thickness ratio)

        edge: a tuple of Vertex objects
        direction : 'horizontal', 'vertical', 'LB_RT', 'LT_RB'
        frame_structure_type : FrameStructureType object
        '''

        # Set edge section corresponding to each frame type and direction
        if direction == 'horizontal' or direction == 'vertical': # chord
            outer_diameter, inward_thickness = frame_structure_type.chord_element_section
        elif direction == 'LB_RT' or direction == 'LT_RB': # brace
            outer_diameter, inward_thickness = frame_structure_type.brace_element_section


        # Add (edge, section tuple) to edge dict overwriting only when outer_diameter, inward_thickness is larger
        if edge in self.edges_dict:
            existing_diameter, existing_thickness = self.edges_dict[edge]
            # calculate area of section of hollow steel tube and compare
            area_existing = existing_diameter**2 - (existing_diameter*(1-existing_thickness))**2
            area_new = outer_diameter**2 - (outer_diameter*(1-inward_thickness))**2
            # apply dominance rule
            if area_new > area_existing:
                self.edges_dict[edge] = outer_diameter, inward_thickness
            # if outer_diameter > existing_diameter or inward_thickness > existing_thickness:
            #     self.edges_dict[edge] = outer_diameter, inward_thickness
        else:
            self.edges_dict[edge] = outer_diameter, inward_thickness


    def _add_external_load(self, vertex):
        '''
        Given that the vertex meets external load, add external load to Vertex object. 
        '''
        ext_load = self.external_load[vertex.coordinates]
        org_load = vertex.load
        updated_load = [a + b for a, b in zip(ext_load, org_load)]
        vertex.load = updated_load

    def get_state(self):
        '''
        Used for when fea graph is directly used as obs
        TODO extract information used as agent observation
        '''
        pass
    
    def get_max_deflection(self):
        '''
         from displacement, get displacement magnitude for each node
            return max displacement node index and magnitude
        '''
        
        # if not hasattr(self, 'displacement') or not self.displacement:
        #     raise ValueError("Displacement data is not available")

        # Calculate the magnitude of displacement for each node
        magnitudes = [np.linalg.norm(node_disp[:2]) for node_disp in self.displacement]

        # Find the index of the node with the maximum displacement
        max_index = np.argmax(magnitudes)
        max_magnitude = magnitudes[max_index]

        return max_index, max_magnitude
    
    def find_coordinates_by_id(self, vertex_id):
        """
        Finds the coordinates of a vertex given its id.
        Args:
            vertex_id (int): The id of the vertex to find.

        Returns:
            tuple: The coordinates of the vertex if found, otherwise None.
        """
        for coordinates, vertex in self.vertices.items():
            if vertex.id == vertex_id:
                return coordinates
        return None  # Return None if no vertex with the given id is found
    
    def get_element_list_with_section(self):
        '''
        Used in pythonASAP-> create_and_solve_model_julia
        Output
            list of (v1_idx, v2_idx, section_outer_diameter, section_inner_wall_thickness_ratio) for each edge in the graph
        '''
        # self.edges_dict : dictionary where keys : (v_1, v_2) Vertex objects, values : (outer diameter, inner wall thickness ratio)
        elem_list_w_section = []
        for v_pairs, section in self.edges_dict.items():
            elem_list_w_section.append((v_pairs[0].id, v_pairs[1].id, section[0], section[1]))

        return elem_list_w_section
    
    def get_element_utilization(self):
        '''
        Used in draw_truss_analysis
        Output
            list of (center_x, center_y, utilization, direction) for each edge in the graph
            where direction is in ['H', 'V', 'D_LB_RT', 'D_LT_RB']
        '''
        res = []
        for i, v_pair in enumerate(self.edges_dict.keys()):
            v1_coords = v_pair[0].coordinates
            v2_coords = v_pair[1].coordinates
            center_x = (v1_coords[0] + v2_coords[0]) / 2
            center_y = (v1_coords[1] + v2_coords[1]) / 2
            util = self.utilization[i]
            # Determine direction based on vertex coordinates
            dx = v2_coords[0] - v1_coords[0]
            dy = v2_coords[1] - v1_coords[1]
            if dy == 0:
                dir = "H"
            elif dx == 0:
                dir = "V"
            elif dx > 0 and dy > 0:
                dir = "D_LB_RT"
            elif dx < 0 and dy > 0: # vertices tored in order of y
                dir = "D_LT_RB"

            res.append((center_x, center_y, util, dir))
        return res



