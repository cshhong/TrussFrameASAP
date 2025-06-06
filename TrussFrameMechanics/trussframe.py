'''
Define TrussFrame object and FrameShapeType
'''

from enum import Enum, auto

class FrameShapeType(Enum):
    '''
    # Checking the values assigned by auto()
    print(FrameShapeType.SQUARE)            # Output: FrameShapeType.SQUARE
    print(FrameShapeType.SQUARE.value)      # Output: 1
    print(FrameShapeType.DIAGONAL_LT_RB)    # Output: FrameShapeType.DIAGONAL_LT_RB
    print(FrameShapeType.DIAGONAL_LT_RB.value)  # Output: 2
    '''
    SQUARE = auto() # assigns an index as value automatically
    DIAGONAL_LT_RB = auto()
    DIAGONAL_LB_RT = auto()
    DOUBLE_DIAGONAL = auto()
    def __str__(self):
        return self.name
    
    @classmethod
    def get_frameshapetype_from_value(cls, value):
        """
        Retrieve the enum member associated with a given value.

        Args:
            value (int): The value to query.

        Returns:
            FrameShapeType: The enum member if found, else raises ValueError.
        """
        for member in cls:
            if member.value == value:
                return member
        # Error handling if value not found
        raise ValueError(f"Invalid FrameShapeType value: {value}")
    
class FrameStructureType(Enum):
    """
    Define type_structure for TrussFrameRL with associated node loads.
    (idx, node_load, is_free_frame)
    load_idx of action is idx value of FrameStructureType
    """
    # EXTERNAL_FORCE = (-1, [0.0, -80.0, 0.0], False)  # Node load in kN
    # UNOCCUPIED = (0, None, False)  # Not used
    # SUPPORT_FRAME = (1, [0.0, -0.4, 0.0], False)  # Node load in kN
    # FST_10_10 = (2, [0.0, -0.4, 0.0], True)  # Node load in kN
    # FST_20_20 = (3, [0.0, -4.0, 0.0], True)  # Node weight in kN
    # Node load in kN

    # EXTERNAL_FORCE = (-1, [0.0, -120.0, 0.0], False)  # Set in generate_bc.py
    # UNOCCUPIED = (0, None, False)  # index used
    # SUPPORT_FRAME = (1, [0.0, -4.0, 0.0], False)  # 0.01m tube with 10% thickness
    # FST_10_10 = (2, [0.0, -4.0, 0.0], True)  # 0.01m tube with 10% thickness
    # FST_20_20 = (3, [0.0, -6.0, 0.0], True)  # 0.01m tube with 20% thickness

    # Define frame types (frame grid idx, node_load, is_free_frame, is_free_nodes, (outer_diameter, inner_wall_thickness_ratio))
    # is_free_nodes are in order of [bottom_left, bottom_right, top_left, top_right] nodes where free nodes are True and fixed nodes (bottom of support frame) are False 
    EXTERNAL_FORCE = ('EXTERNAL_FORCE', None, 20, [0.0, -120.0, 0.0], False, None, None)  # Set in generate_bc.py
    UNOCCUPIED = ('UNOCCUPIED',None, 0, None, False, None, None)  # index used
    # SUPPORT_FRAME = ('SUPPORT_FRAME', 1, [0.0, -10.0, 0.0], False, (False, False, True, True), ((0.2, 0.4),(0.2, 0.4)))  # chord/brace 0.01m tube with 10% thickness
    SUPPORT_FRAME = ('SUPPORT_FRAME', FrameShapeType.DOUBLE_DIAGONAL, 10, [0.0, -8.0, 0.0], False, (False, False, True, True), ((0.2, 0.2),(0.2, 0.2)))  # chord/brace 0.01m tube with 10% thickness
    FST_c10_10_b10_10_d = ('FST_c10_10_b10_10_d', FrameShapeType.DOUBLE_DIAGONAL, 2, [0.0, -4.0, 0.0], True, (True, True, True, True), ((0.1, 0.1),(0.1, 0.1)))  # 0.01m tube with 10% thickness
    # FST_c10_10_b10_10_ltrb = ('FST_c10_10_b10_10_ltrb', FrameShapeType.DIAGONAL_LT_RB, 3, [0.0, -3.5, 0.0], True, (True, True, True, True), ((0.1, 0.1),(0.1, 0.1)))  # 0.01m tube with 10% thickness
    # FST_c10_10_b10_10_rtlb = ('FST_c10_10_b10_10_rtlb', FrameShapeType.DIAGONAL_LB_RT, 4, [0.0, -3.5, 0.0], True, (True, True, True, True), ((0.1, 0.1),(0.1, 0.1)))  # 0.01m tube with 10% thickness

    # # FST_c10_10_b20_20 = ('FST_c10_10_b20_20', 3, [0.0, -6.0, 0.0], True, (True, True, True, True),  ((0.1, 0.1),(0.2, 0.2)),)  # 0.01m tube with 20% thickness
    # FST_c10_15_b10_15_d = ('FST_c10_15_b10_15_d', FrameShapeType.DOUBLE_DIAGONAL, 3, [0.0, -5.0, 0.0], True, (True, True, True, True),  ((0.1, 0.15),(0.1, 0.15)),)  # 0.01m tube with 20% thickness
    FST_c10_25_b10_25_d = ('FST_c10_25_b10_25_d', FrameShapeType.DOUBLE_DIAGONAL, 3, [0.0, -6.0, 0.0], True, (True, True, True, True),  ((0.1, 0.25),(0.1, 0.25)),)  # 0.01m tube with 20% thickness
    FST_c20_10_b20_10_d = ('FST_c20_10_b20_10_d', FrameShapeType.DOUBLE_DIAGONAL, 4, [0.0, -6.0, 0.0], True, (True, True, True, True),  ((0.2, 0.1),(0.2, 0.1)),)  # 0.01m tube with 20% thickness

    # May26 add
    # FST_c20_20_b20_20_d = ('FST_c20_20_b20_20_d', FrameShapeType.DOUBLE_DIAGONAL, 4, [0.0, -8.0, 0.0], True, (True, True, True, True),  ((0.2, 0.2),(0.2, 0.2)),) 

    # set class variable
    default_type = FST_c10_10_b10_10_d

    def __init__(self, name_str, shape_type, idx, node_load, is_free_frame, is_free_nodes, element_section):
        self.name_str = name_str
        self.idx = idx
        self.shape_type = shape_type  # FrameShapeType
        self._node_load = node_load # tuple of 4 (x, y, z) loads in kN for (bottom_left, bottom_right, top_left, top_right)
        self.is_free_frame = is_free_frame # boolean value for free frame
        self.is_free_nodes = is_free_nodes # boolean values for (bottom_left, bottom_right, top_left, top_right)
        self.chord_element_section = element_section[0] if element_section!=None else None # (outer_diameter, inner_wall_thickness_ratio)
        self.brace_element_section = element_section[1] if element_section!=None else None# (outer_diameter, inner_wall_thickness_ratio)

    @property
    def node_load(self):
        return self._node_load
    
    @node_load.setter
    def node_load(self, value):
        """
        Allow node_load to be set only for the FORCE member.
        """
        if self is FrameStructureType.EXTERNAL_FORCE:
            self._node_load = value
        else:
            raise AttributeError(f"Cannot modify node_load for {self.name}")
        
    @classmethod
    def get_node_load_from_idx(cls, idx):
        """
        Retrieve the node_load associated with a given idx value.
        Args:
            idx (int): The idx value to query.
        Returns:
            tuple or None: The node_load tuple if found, else None.
        """
        for member in cls:
            if member.idx == idx:
                return member.node_load
        return None  # Or raise an exception if idx not found
    
    @classmethod
    def get_framestructuretype_from_idx(cls, idx):
        """
        Retrieve the enum member associated with a given idx value.
        
        Args:
            idx (int): The idx value to query.

        Returns:
            FrameStructureType or None: The enum member if found, else None.
        """
        for member in cls:
            if member.idx == idx:
                return member
        # Error handling if idx not found
        raise ValueError(f"Invalid FrameStructrueType idx value: {idx}")
    
    @classmethod
    def get_free_frame_types(cls):
        """
        Retrieve a list of all free frame types.
        Returns:
            list of FrameStructureType: Members that are free frame types.
        """
        return [member for member in cls if member.is_free_frame]
    
    @classmethod
    def get_freeframe_idx_bounds(cls):
        """
        Count the number of free frame types.
        Returns:
            int: The number of free frame types.
        """
        freeframe_idx_values = [member.idx for member in cls if member.is_free_frame]
        return min(freeframe_idx_values), max(freeframe_idx_values)
    
    @classmethod
    def get_idx_bounds(cls):
        idx_values = [member.idx for member in cls]
        return min(idx_values), max(idx_values)
    
    
class TrussFrameRL:
    '''
    used in cantileverenv_v0.py
    TrussFrame object with frame centroid position and frame type
    coordinate is in absolute coordinates of the board (not relative to support)
    By Default is set to type with FrameShapeType.DIAGONAL_LT_RB(diagonal brace from left top to right bottom)

    Can also be placeholder for support and target load
    '''
    _id_counter = 0  # Class-level counter for automatically assigning unique IDs

    def __init__(self, pos, frame_size=2, type_structure=FrameStructureType.default_type):
        # Frame centroid coordinate on board
        self.x = pos[0] 
        self.y = pos[1] 
        self.bottom_left = (self.x - frame_size/2, self.y - frame_size/2) # bottom left corner of the frame
        self.bottom_right = (self.x + frame_size/2, self.y - frame_size/2) # bottom right corner of the frame
        self.top_left = (self.x - frame_size/2, self.y + frame_size/2) # top left corner of the frame   
        self.top_right = (self.x + frame_size/2, self.y + frame_size/2) # top right corner of the frame

        # Frame cell position on frame grid
        self.x_frame = int(self.x // frame_size)
        self.y_frame = int(self.y // frame_size)
        self.type_structure = type_structure # FrameStructureType
        self.type_shape = type_structure.shape_type  # FrameShapeType
        
        # Automatically count and assign a unique ID - support, target load, following frames
        self.id = TrussFrameRL._id_counter
        TrussFrameRL._id_counter += 1  # Increment the class-level counter

    def get_vertex_positions(self):
        """
        Get the positions of the vertices of the frame in order of bottom_left, bottom_right, top_left, top_right.
        Returns:
            list: A list of (board_x, board_y) tuples representing the positions of the vertices.
        """
        return [self.bottom_left, self.bottom_right, self.top_left, self.top_right]