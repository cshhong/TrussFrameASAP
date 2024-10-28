'''
Functions to generate boundary conditions (support, external loads) within the environment
'''
import random
from TrussFrameMechanics.trussframe import FrameStructureType

CELL_SIZE = 1
FRAME_SIZE = 2

def set_cantilever_env(board_size_x, square_size, seed=None):
    """
    Used in main.py (human playable setting)
    Set up the cantilever environment with parametric boundary conditions.

    Input:
    - square_size: Size of each square in the grid

    Output:
    - pinned_supports: Dictionary where key is the node idx i, and value is (x, y) coordinates representing the pinned supports
                       (coordinates are on a square_size grid, spaced 2 units apart horizontally)
    - target_load: Dictionary where key is the (x, y) grid coordinate and value is the [x, y, z] force magnitude
                   (coordinates are on the square_size grid, force magnitude is applied in negative y direction)
    """

    # If a seed is provided, set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Sample boundary conditions
    num_supports_options = [1,2]
    height_options = [1, 2]
    length_options = [3, 4, 5]
    magnitude_options = [10, 20, 30]

    # Choose random height, length, and load magnitude
    height = random.choice(height_options)
    length = random.choice(length_options)
    magnitude = random.choice(magnitude_options)
    num_supports = random.choice(num_supports_options)

    # Set pinned supports
    # x at center even coordinate of board
    x_support_start = (board_size_x // 2 - 2) if (board_size_x // 2 - 2) % 2 == 0 else (board_size_x // 2 - 3)
    y_support = square_size  # Base of the cantilever

    # Create two pinned supports self.square_size units apart on the grid
    # indices are 1,2 for the first support, 5,6 for the second support 
    # first support is placed starting x_support_start 
    # second support is placed starting x_support_start - length * square_size
    # First support
    if num_supports == 1:
        support_nodes = {
            1: (x_support_start, y_support),
            2: (x_support_start + square_size, y_support)
        }
        # Get one / two default TrussFrame coordinates (x,y) center of frame
        default_frames = [(x_support_start+1, y_support+1)]

    # If 2 supports, add the second support to the dictionary
    if num_supports == 2:
        x_support_start_second = x_support_start - (length - 1) * square_size
        support_nodes = {
            1: (x_support_start, y_support),
            2: (x_support_start + square_size, y_support),
            5: (x_support_start_second, y_support),
            6: (x_support_start_second + square_size, y_support),
        }
        # Get one / two default TrussFrame coordinates (x,y) center of frame
        default_frames = [(x_support_start+1, y_support+1) , (x_support_start_second+1, y_support+1)]
    

    # Set target load: Applied at a point away from the pinned supports
    load_x = x_support_start + length * square_size
    load_y = y_support + (height + 1) * square_size

    # Target load in negative y direction (z assumed to be zero)
    target_load = {
        (load_x, load_y): [0, -magnitude, 0]
    }

    return default_frames, support_nodes, target_load


def set_cantilever_env_framegrid(frame_grid_size_x, seed=None):
    """
    Used in cantileverenv_V0.py (agent playable setting)
    Set up the cantilever environment within the frame grid with parametric boundary conditions.
                    y
                     ↑
                     •----•----•----•----•----•----•
                     |    |    |    |    |    |    |
                2    •-  -0-  -|-  -0-  -|- -(1)- -|
                     |    |    |    |    |    |    |
                     •----•----•----•----•----•----•
                     |    |    |    |    |    |    |
                1    •-  -1-  -|-  -1-  -|-  -0-  -|
                     |    |    |    |    |    |    |
                     •----•----•----•----•----•----•
                     |    |    |    |    |    |    |
                0    •-  -0-  -|-  -2-  -|-  -0-  -|
                     |    |    |    |    |    |    | 
                     •----•----•----•----•----•----•
                          0         1         2 

    Create one support frame and one external load that is height & length to the right side of the support

    Input:
        - frame_grid_size_x: Number of frames along the x-dimension of the frame grid.
    
    Output:
        - support_frames : list of tuples (x_frame, y_frame) 
        - targetload_frames : dictionary ((x_frame,y_frame) : [x_forcemag, y_forcemag, z_forcemag]) tuples (force is applied in the negative y direction).
        - inventory : dictionary of FrameStructureType Free frame type : count 
    """

    # If a seed is provided, set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Sample boundary conditions
    # height_options = [1, 2, 3]                 # in terms of frames
    height_options = [1, ]                 # in terms of frames
    # length_options = [3, 4, 5]              # in terms of frames
    length_options = [3,]              # in terms of frames
    # magnitude_options = [10.0, 20.0, 30.0, 40.0]    # kN
    magnitude_options = [40.0]    # kN


    # Choose random height, length, and load magnitude
    height = random.choice(height_options)
    length = random.choice(length_options)
    magnitude = random.choice(magnitude_options)
    med_inv = random.choice(range(1,length)) # size of medium free frame inventory ; set to length of cantilever
    med_inv = 2

    inventory = {
        FrameStructureType.LIGHT_FREE_FRAME : -1, # indicate no limits
        FrameStructureType.MEDIUM_FREE_FRAME : med_inv,
        # FrameStructureType.HEAVY_FREE_FRAME : *,
    }

    # Set pinned supports within the frame grid
    # x at center even coordinate of the frame grid
    x_support_start_frame = (frame_grid_size_x // FRAME_SIZE - (FRAME_SIZE//2)) if (frame_grid_size_x // FRAME_SIZE - (FRAME_SIZE//2)) % FRAME_SIZE == 0 else (frame_grid_size_x // FRAME_SIZE - FRAME_SIZE)
    y_support_frame = 0  # Base of the cantilever, first row of the frame grid

    # Create 1 support frame
    support_frames = [
            (x_support_start_frame, y_support_frame),  # Support frame at (x, y)
        ]

    # Set target load frame within the frame grid
    load_x_frame = x_support_start_frame + length
    load_y_frame = y_support_frame + height

    # Target load in negative y direction (z assumed to be zero)
    target_frames = {
        (load_x_frame, load_y_frame): [0, -magnitude, 0]
    }

    return support_frames, target_frames, inventory