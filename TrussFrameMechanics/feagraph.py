"""
Custom graph data structure that stores additional information needed for structural analysis
"""
from .maximaledge import MaximalEdge  # Relative import
from .trussframe import FrameType     # Relative import

    
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
    edges : An adjacency list of tuples representing edges, where each tuple contains vertex indices.
    maximal_edges : A dictionary where keys are directions and values are a list of MaximalEdge objects. (only useful for overlapping frames)
    external_loads : dictionary where key : coordinate in board, value : load magnitude [load.x, load.y, load.z]
    displacement :  2D list of nodal displacement [x,y,z] for each node in node index order. Only non-empty upon fea (eps end)
    
    """
    
    def __init__(self, vertices=None, supports=None, edges=None, maximal_edges=None, external_loads=None, default_node_load=[0.0, -0.4, 0.0]):
        """
        Initializes the graph with dictionaries of vertices, edges, and maximal edges.
        """

        self.default_node_load = default_node_load # kN
        # Decided with epside env init
        self.supports = supports if supports is not None else []
        self.external_loads = external_loads if external_loads is not None else {}
        
        self.vertices = vertices if vertices is not None else {}
        self.edges = edges if edges is not None else []
        self.maximal_edges = maximal_edges if maximal_edges is not None else {
                                                                                'horizontal': [],
                                                                                'vertical': [],
                                                                                'LB_RT': [], 
                                                                                'LT_RB': []
                                                                            }
        self.maximal_edges_merged = False
        self.displacement = []

    def __repr__(self):
        """Nicely formatted representation of the graph."""
        supports_repr = "\n".join([f"  {s}" for s in self.supports])
        externalloads_repr = "\n".join([f"  {coord}: {load_val}" for coord, load_val in self.external_loads.items()])
        vertices_repr = "\n".join([f"  {coord}: {v}" for coord, v in self.vertices.items()])
        edges_repr = "\n".join([f"  {e}" for e in self.edges])
        maximal_edges_repr = "\n".join([f"  {d}: {edges}" for d, edges in self.maximal_edges.items()])
        displacement_repr = "\n".join([f"  {node_idx}: {displacement}" for node_idx, displacement in enumerate(self.displacement)])

        return (
            f"Default Node load : ({self.default_node_load})\n"
            f"Supports ({len(self.supports)}):\n{supports_repr}\n"
            f"External Loads ({len(self.external_loads)}):\n{externalloads_repr}\n"
            # f"Vertices ({len(self.vertices)}):\n{vertices_repr}\n"
            f"Vertices ({len(self.vertices)})\n"
            # f"Edges ({len(self.edges)}):\n{edges_repr}\n"
            f"Edges ({len(self.edges)})\n"
            # f"Maximal Edges ({len(self.maximal_edges)}):\n{maximal_edges_repr}\n"
            f"Displacement ({len(self.displacement)}):\n{displacement_repr}\n)"
        )

    def get_all_node_ids(self):
        """Get all integer value node ids from FEAGraph object"""
        return [vertex.id for vertex in self.vertices.values()]
    
    def combine_and_merge_edges(self, frame_type_shape, new_vertices):
        '''
        Input
            frame_type : FrameType object to indicate which vertices should be connected
            new_vertices : List of Vertex objects in order of bottom-left, bottom-right, top-right, top-left
        Check overlapping edge segments and merge with maximal edge representation
        Update self.curr_fea_graph.edges, self.curr_fea_graph.maximal_edges in place
        used within update_curr_fea_graph
        '''
        # check horizontal edges 
        h1 = (new_vertices[0], new_vertices[1]) # Vertex objects 
        h2 = (new_vertices[3], new_vertices[2])
        # check vertical edges
        v1 = (new_vertices[1], new_vertices[2])
        v2 = (new_vertices[0], new_vertices[3])
        # check diagonal lines
        d1, d2 = None, None
        if frame_type_shape == FrameType.DIAGONAL_LB_RT:
            d1 = (new_vertices[0], new_vertices[2])
        elif frame_type_shape == FrameType.DIAGONAL_LT_RB:
            d2 = (new_vertices[1], new_vertices[3])
        elif frame_type_shape == FrameType.DOUBLE_DIAGONAL:
            d1 = (new_vertices[0], new_vertices[2])
            d2 = (new_vertices[1], new_vertices[3])
        segments = {
                    'horizontal': [h1, h2],
                    'vertical': [v1, v2],
                    'LB_RT': [d1],
                    'LT_RB': [d2]
                }
        for direction, segs in segments.items():
            self._combine_segments(segs, direction)

        # Update edge list with new line segments
        # get minimal edge list from each maximal edge
        maximal_edges = self.maximal_edges
        all_edges = []
        for dir in maximal_edges:
            self._merge_maximal_edges() # update maximal edges that are merged from new frame 
            for me in maximal_edges[dir]:
                # print(f'extracting edge from maximal edge : {me}')
                all_edges.extend(me.get_edge_list()) # get list of tuples of vertex indices
        self.edges = all_edges
    
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

        self.maximal_edges_merged = False
    
    def _merge_maximal_edges(self):
        '''
        After combining segment, merge maximal edges that are connected from new frame 
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

        self.maximal_edges_merged = True

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

    


