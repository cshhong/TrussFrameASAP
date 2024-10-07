"""
Custom graph data structure that stores additional information needed for structural analysis
"""
from maximaledge import MaximalEdge
    
class FEAGraph:
    """
    A data structure representing an immutable graph that consists of vertices,
    edges, and maximal edges. Once created, the graph cannot be modified.
    
    Attributes:
    -----------
        
    vertices : dict
        A dictionary where keys are coordinates and values are Vertex objects (read-only).
        
    supports : list
        list of vertex idx where the nodes in the frame are supports / pinned (as opposed to free)

    edges : list
        An adjacency list of tuples representing edges, where each tuple contains
        vertex indices (read-only).
    
    maximal_edges : dict
        A dictionary of maximal edges grouped by direction. Each direction contains
        a list of MaximalEdge objects (read-only).

    loads : list
        A list of tuples (node.id, [load.x, load.y, load.z]) 
        [
            (4, [-4000.0, -8000.0, 0.0]),  
            (3, [-4000.0, -8000.0, 0.0]), 
        ]
    
    """
    
    def __init__(self, vertices=None, supports=None, edges=None, maximal_edges=None, loads=None):
        """
        Initializes the graph with dictionaries of vertices, edges, and maximal edges.
        After initialization, the graph becomes immutable.
        """

        # Create deep copies of the inputs to ensure no external mutations
        self.vertices = vertices if vertices is not None else {}
        self.supports = supports if supports is not None else []
        self.edges = edges if edges is not None else []
        self.maximal_edges = maximal_edges if maximal_edges is not None else {}
        self.maximal_edges_merged = False
        self.loads = loads if loads is not None else []

        # # Create deep copies of the inputs to ensure no external mutations
        # self._vertices = vertices if vertices is not None else {}
        # self._supports = supports if supports is not None else []
        # self._edges = edges if edges is not None else []
        # self._maximal_edges = maximal_edges if maximal_edges is not None else {}
        # self._loads = loads if loads is not None else []
    
    # @property
    # def vertices(self):
    #     """Read-only access to the vertices."""
    #     return self._vertices
    
    # @property
    # def supports(self):
    #     """Read-only access to the supports."""
    #     return self._supports
    
    # @property
    # def edges(self):
    #     """Read-only access to the edges."""
    #     return self._edges
    
    # @property
    # def maximal_edges(self):
    #     """Read-only access to the maximal edges."""
    #     return self._maximal_edges
    
    # @property
    # def loads(self):
    #     """Read-only access to the loads."""
    #     return self._loads
    
    def __repr__(self):
        """Nicely formatted representation of the graph."""
        vertices_repr = "\n".join([f"  {coord}: {v}" for coord, v in self.vertices.items()])
        supports_repr = "\n".join([f"  {s}" for s in self.supports])
        edges_repr = "\n".join([f"  {e}" for e in self.edges])
        maximal_edges_repr = "\n".join([f"  {d}: {edges}" for d, edges in self.maximal_edges.items()])
        loads_repr = "\n".join([f"  Node {node_id}: Load {load}" for node_id, load in self.loads])
        
        return (
            f"Vertices ({len(self.vertices)}):\n{vertices_repr}\n\n"
            f"Supports ({len(self.supports)}):\n{supports_repr}\n\n"
            f"Edges ({len(self.edges)}):\n{edges_repr}\n\n"
            f"Maximal Edges ({len(self.maximal_edges)}):\n{maximal_edges_repr}\n\n"
            f"Loads ({len(self.loads)}):\n{loads_repr}"
        )

    def get_all_node_ids(self):
        """Get all integer value node ids from FEAGraph object"""
        return [vertex.id for vertex in self.vertices.values()]
    
    def combine_segments(self, segments, direction):
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
    
    def merge_maximal_edges(self):
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


