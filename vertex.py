"""
Custom Vertex data structure that stores id, coordinate, edge information
"""

from functools import total_ordering

@total_ordering
class Vertex:
    '''
    Vertex in the graph that represents a node in the truss frame.
        coordinates: A tuple representing the position of the vertex (e.g., (x, y) for 2D space).
        id: A unique integer identifier for each vertex, automatically assigned using a class-level counter (_id_counter).
        edges: A list of edges connected to this vertex, which can store Edge objects. (unused) #TODO
    '''
    _id_counter = 1  # Class-level counter for vertex IDs

    def __init__(self, coordinates):
        self.coordinates = coordinates
        # self.id = f'v{Vertex._id_counter}'
        self.id = Vertex._id_counter
        Vertex._id_counter += 1
        self.edges = []

    def __repr__(self):
        # return f"Vertex({self.id}, {self.coordinates})"
        return f"Vertex({self.id})"
    
    def __lt__(self, other):
        '''
        Vertex order is based on coordinates from left to right, bottom to top
        for vertices in direction horizontal, vertical, LB_RT, LT_RB
        '''
        # vertical order
        if self.coordinates[0] == other.coordinates[0]:
            return self.coordinates[1] < other.coordinates[1]
        else:
            return self.coordinates[0] < other.coordinates[0]
        # else:
        #     raise ValueError(f"Cannot compare vertices {self.coordinates} and {other.coordinates} with different x and y coordinates.")
    
    def __eq__(self, other):
        '''
        Checks whether two vertices are equal based on their coordinates.
        '''
        assert isinstance(other, Vertex), "Can only compare Vertex objects."
        if isinstance(other, Vertex):
            return self.coordinates == other.coordinates
        return False
    
    def __hash__(self):
        return hash((self.id, self.coordinates))