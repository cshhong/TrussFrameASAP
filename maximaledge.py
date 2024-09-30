"""
Used to process TrussFrame -> Vertices and Edges in a Graph structure
    - handles overlapping vertices, overlapping and extended edges by organizing in maximal graph

Also used to make visualization efficient (maximal edge is drawn at once)
"""
# from utils import are_segments_colinear

class MaximalEdge:
    '''
    Represents a maximal edge in the graph. A maximal edge is a set of connected vertices in a specific direction 
    (such as horizontal, vertical, or diagonal). 
    It could span multiple smaller edges or segments.

    Purpose: In this implementation, there is no explicit Edge class, but edges are dynamically extracted from maximal edges. 
    The edge list stores individual line segments (edges) that connect two vertices.
    
    '''
    # _id_counter = 0  # Class-level counter for edge IDs

    def __init__(self, direction):
        '''
        vertices : list of Vertex objects that define the endpoints of the maximal edge.
        direction : str ('horizontal', 'vertical', 'LB_RT', 'LT_RB')
        '''
        self.direction = direction
        self.vertices = []  # list of Vertex objects indices that are updated as truss frames are added
        # self.id = f'e{MaximalEdge._id_counter}'
        # self.pos = (self.vertices[0].coordinates[0] + self.vertices[-1].coordinates[0]) / 2, (self.vertices[0].coordinates[1] + self.vertices[-1].coordinates[1]) / 2
        # MaximalEdge._id_counter += 1

    def __repr__(self):
        return f"MaximalEdge({self.direction}, {self.vertices})"

     
    def merge_segment(self, start_vertex, end_vertex):
        """
        Given new line segment that has same direction as the current maximal edge, merge the new line into the current maximal edge
        Input 
            start_vertex, end_vertex : Vertex objects

        The function:
        - Checks overlap with the current maximal edge. (colinear and within the bounds of the maximal edge)
        - Merges the new edge if appropriate.
        - Extends the maximal edge if necessary.
        - Returns True if merged, False if no overlap.
        
        Merge cases 
            case 1 : overlap with maximal edge extends and existing node - do nothing
            case 2 : overlap with maximal edge extends and new node - add new node within maximal edge
            case 3 : extend maximal edge with new node - add new node within maximal edge, extend maximal edge
            case 4 : extend maximal edge overlap with existing node - extend maximal edge
            case 5 : no overlap - do nothing 
        
        """
        # print(f'merging {start_vertex} and {end_vertex} to edge {self.vertices}')
        # check if new edge is in the same direction
        dir = self._check_edge_direction(start_vertex, end_vertex)
        assert self.direction == dir , f'edge direction {self.direction} is not the same as new edge direction {dir}'
        # check if new edge is colinear with the current maximal edge
        if are_segments_colinear((start_vertex.coordinates, end_vertex.coordinates),
                                 (self.vertices[0].coordinates, self.vertices[-1].coordinates)):
            # Case 1, 2, 3, 4: Merge into the current maximal edge
            if start_vertex <= self.vertices[-1] and end_vertex >= self.vertices[0]: # check within bounds
                # add vertices to the current maximal edge
                self.vertices = sorted(set(self.vertices + [start_vertex, end_vertex]))
                # print(f'merged! {self.vertices}')
                return True # updated self
        
        # Case 5: No overlap
        return False # No merge occurred, needs to create a new maximal edge

    def get_edge_list(self):
        '''
        return minimal edge list as tuple of Vertex ids from maximal edge
        '''
        edge_list = []
        for i in range(len(self.vertices) - 1):
            edge_list.append((self.vertices[i].id, self.vertices[i + 1].id))
        return edge_list

    def _check_edge_direction(self, start_vertex, end_vertex):
        '''
        return str ('horizontal', 'vertical', 'LB_RT', 'LT_RB') based on the direction of the edge
        '''
        
        if (start_vertex.coordinates[0] == end_vertex.coordinates[0]) and start_vertex.coordinates[1] == end_vertex.coordinates[1]:
            raise ValueError(f"edge {start_vertex.coordinates} and {end_vertex.coordinates} are the same vertex.")
        elif (start_vertex.coordinates[0] == end_vertex.coordinates[0]) and start_vertex.coordinates[1] != end_vertex.coordinates[1]:
            return 'vertical'
        elif (start_vertex.coordinates[1] == end_vertex.coordinates[1]) and start_vertex.coordinates[0] != end_vertex.coordinates[0]:
            return 'horizontal'
        else:
            slope = (end_vertex.coordinates[1] - start_vertex.coordinates[1]) / (end_vertex.coordinates[0] - start_vertex.coordinates[0])
            if end_vertex.coordinates[0] - start_vertex.coordinates[0] == 0:
                raise ValueError(f"edge {start_vertex.coordinates} and {end_vertex.coordinates} have same x value")
            elif slope == 1: # check if slope is 1 
                return 'LB_RT'
            elif slope == -1:   # check if slope is -1
                return 'LT_RB'
            else:
                raise ValueError(f"edge {start_vertex.coordinates} and {end_vertex.coordinates} does not have direction horizontal, vertical, LB_RT, LT_RB.")
        
    def is_connected(self, other):
        '''
        Check if two maximal edges are connected
        Input
            other : MaximalEdge object
        '''
        # assert self.direction == other.direction, f"Maximal edges {self} and {other} are not in the same direction."
        # check colinearity
        if are_segments_colinear((self.vertices[0].coordinates, self.vertices[-1].coordinates),
                                (other.vertices[0].coordinates, other.vertices[-1].coordinates)):
            # case 1 : self succeeds other
            if (self.vertices[0] <= other.vertices[-1]) and (self.vertices[0] >= other.vertices[0]):
                return True
            # case 2 : self precedes other
            if (self.vertices[-1] >= other.vertices[0]) and (self.vertices[-1] <= other.vertices[-1]):
                return True
        return False
    
    def add_vertex(self, vertex):
        """Add a vertex to the maximal edge."""
        # insert the vertex in the correct position
        self.vertices.append(vertex)
        self.vertices.sort()
        # adjust the position value of the vertex
        self.pos = (self.vertices[0].coordinates[0] + self.vertices[-1].coordinates[0]) / 2, (self.vertices[0].coordinates[1] + self.vertices[-1].coordinates[1]) / 2



'''
Geometric functions 
'''

def are_colinear(p1, p2, p3):
        """
        Check if three points are collinear.
        Returns True if collinear, otherwise False.
        """
        (x1, y1) = p1
        (x2, y2) = p2
        (x3, y3) = p3
        
        # Calculate the area of the triangle formed by p1, p2, and p3
        area = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        
        # If the area is 0, the points are collinear
        return area == 0


def are_segments_colinear(segment1, segment2):
    """
    Check if two line segments are collinear.
    Each segment is represented as ((x1, y1), (x2, y2)).
    """
    (p1, p2) = segment1
    (p3, p4) = segment2
    
    # Check collinearity of (p1, p2, p3) and (p1, p2, p4)
    return are_colinear(p1, p2, p3) and are_colinear(p1, p2, p4)
