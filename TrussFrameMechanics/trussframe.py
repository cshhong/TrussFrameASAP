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
    # LIGHT_FREE_FRAME = (2, [0.0, -0.4, 0.0], True)  # Node load in kN
    # MEDIUM_FREE_FRAME = (3, [0.0, -4.0, 0.0], True)  # Node weight in kN
    # Node load in kN
    EXTERNAL_FORCE = (-1, [0.0, -120.0, 0.0], False)  # Set in generate_bc.py
    UNOCCUPIED = (0, None, False)  # index used
    SUPPORT_FRAME = (1, [0.0, -4.0, 0.0], False)  # 0.01m tube with 10% thickness
    LIGHT_FREE_FRAME = (2, [0.0, -4.0, 0.0], True)  # 0.01m tube with 10% thickness
    MEDIUM_FREE_FRAME = (3, [0.0, -6.0, 0.0], True)  # 0.01m tube with 20% thickness

    def __init__(self, idx, node_load, is_free_frame):
        self.idx = idx
        self._node_load = node_load
        self.is_free_frame = is_free_frame

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
    
    

class TrussFrame:
    '''
    used in main.py
    TrussFrame object with centroid position and frame type
    By Default is set to type with FrameShapeType.DIAGONAL_LT_RB(diagonal brace from left top to right bottom)
    
    '''
    def __init__(self, pos, type_shape=FrameShapeType.DOUBLE_DIAGONAL):
        self.x = pos[0] # local (x,y) center grid position of the frame
        self.y = pos[1] # local (x,y) center grid position of the frame
        self.type_shape = type_shape  # Type of the frame

class TrussFrameRL:
    '''
    used in cantileverenv_v0.py
    TrussFrame object with frame centroid position and frame type
    coordinate is in absolute coordinates of the board (not relative to support)
    By Default is set to type with FrameShapeType.DIAGONAL_LT_RB(diagonal brace from left top to right bottom)

    Can also be placeholder for support and target load
    '''
    _id_counter = 0  # Class-level counter for automatically assigning unique IDs

    def __init__(self, pos, type_shape=FrameShapeType.DOUBLE_DIAGONAL, frame_size=2, type_structure=FrameStructureType.LIGHT_FREE_FRAME):
        self.type_shape = type_shape  # FrameShapeType
        # Frame centroid coordinate on board
        self.x = pos[0] 
        self.y = pos[1] 
        # Frame cell position on frame grid
        self.x_frame = int(self.x // frame_size)
        self.y_frame = int(self.y // frame_size)
        self.type_structure = type_structure # FrameStructureType
        
        # Automatically count and assign a unique ID - support, target load, following frames
        self.id = TrussFrameRL._id_counter
        TrussFrameRL._id_counter += 1  # Increment the class-level counter

    # def get_relative_pos(self, ref_pos):
    #     '''
    #     Get position relative to the position of a reference cell (support, external load)
    #     Used to generate edge values within frame graph 

    #     Input:
    #         ref_pos : (x,y) position of reference cell (support, external load)
    #     '''
    #     rel_x = ref_pos[0] - self.x
    #     rel_y = ref_pos[1] - self.y

    #     return rel_x, rel_y