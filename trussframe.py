'''
Define TrussFrame object and FrameType
'''

from enum import Enum, auto

class FrameType(Enum):
    '''
    # Checking the values assigned by auto()
    print(FrameType.SQUARE)            # Output: FrameType.SQUARE
    print(FrameType.SQUARE.value)      # Output: 1
    print(FrameType.DIAGONAL_LT_RB)    # Output: FrameType.DIAGONAL_LT_RB
    print(FrameType.DIAGONAL_LT_RB.value)  # Output: 2
    '''
    SQUARE = auto() # assigns an index as value automatically
    DIAGONAL_LT_RB = auto()
    DIAGONAL_LB_RT = auto()
    DOUBLE_DIAGONAL = auto()
    def __str__(self):
        return self.name

class TrussFrame:
    '''
    TrussFrame object with centroid position and frame type
    By Default is set to type with FrameType.DIAGONAL_LT_RB(diagonal brace from left top to right bottom)
    
    '''
    def __init__(self, pos, type_shape=FrameType.DIAGONAL_LT_RB):
        self.x = pos[0] # local (x,y) center grid position of the frame
        self.y = pos[1] # local (x,y) center grid position of the frame
        self.type_shape = type_shape  # Type of the frame

class TrussFrameRL:
    '''
    TrussFrame object with centroid position and frame type
    coordinate is in absolute coordinates of the board (not relative to support)
    By Default is set to type with FrameType.DIAGONAL_LT_RB(diagonal brace from left top to right bottom)

    Can also be placeholder for support and target load
    '''
    _id_counter = 0  # Class-level counter for automatically assigning unique IDs

    def __init__(self, pos, type=1, type_shape=FrameType.DIAGONAL_LT_RB, frame_size=2, type_structure=-1):
        self.type_shape = type_shape  # Type of the frame
        self.x = pos[0] 
        self.y = pos[1] 
        self.x_frame = int(self.x // frame_size)
        self.y_frame = int(self.y // frame_size)
        self.type_structure = type # free frame = 1, support frame = 2, force = -1
        
        # Automatically count and assign a unique ID
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