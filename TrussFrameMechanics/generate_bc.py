'''
Functions to generate boundary conditions (support, external loads) within the environment
'''
import random
# from TrussFrameASAP.TrussFrameMechanics.trussframe import FrameStructureType
from TrussFrameMechanics.trussframe import FrameStructureType

CELL_SIZE = 1
FRAME_SIZE = 2 # one truss frame is 2x2 cells

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

def set_cantilever_env_framegrid(
        frame_grid_size_x, 
        height_options = [1, 2, 3],
        length_options = [3, 4, 5],
        magnitude_options = [300, 400, 500],
        inventory_options = [(10,10), (10,5), (5,5), (8,3)], 
        seed=None):
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
    # height_options = [1,]                 # in terms of frames
    # length_options = [3, 4, 5]              # in terms of frames
    # length_options = [3, 5]              # in terms of frames
    # length_options = [5,]              # in terms of frames
    # magnitude_options = [10.0, 20.0, 30.0, 40.0]    # kN
    # magnitude_options = [120.0]    # kN
    # magnitude_options = [200.0, 300.0, 400.0]    # kN 300, 
    # magnitude_options = [300.0,]    # kN 300, 


    # light_inv = frame_grid_size_x * 2 # TODO reasonable inventory for light frame?
    # med_inv = random.choice(range(1,length)) # size of medium free frame inventory ; set to length of cantilever
    
    # inventory_options = [(15,10), (10,5), (8,3)]
    # inventory_options = [(10,10), (10,5), (5,5), (8,3)]
    # inventory_options = [(7,7)]
    light_inv, med_inv = random.choice(inventory_options)
    # light_inv = 15
    # med_inv = 10

    inventory = {
        FrameStructureType.FST_10_10 : light_inv, # -1 indicate no limits
        FrameStructureType.FST_20_20 : med_inv,
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

    # Choose random height, length, and load magnitude
    height = random.choice(height_options)
    length = random.choice(length_options)
    magnitude = random.choice(magnitude_options)

    # Set target load frame within the frame grid
    load_x_frame = x_support_start_frame + length
    load_y_frame = y_support_frame + height
    # Target load in negative y direction (z assumed to be zero)
    target_frames = {
        (load_x_frame, load_y_frame): [0, -magnitude, 0]
    }

    
    # # Set multiple target load frames
    # target_frames = dict()
    # # one on right and one on left
    # # randomly choose from height, length, magnitude options for each side
    # for i in range(1,2):
    #     # Choose random height, length, and load magnitude
    #     height = random.choice(height_options)
    #     length = random.choice(length_options)
    #     magnitude = random.choice(magnitude_options)

    #     # Set target load frame within the frame grid
    #     load_x_frame = x_support_start_frame + (-1)^i * length
    #     load_y_frame = y_support_frame + height

    #     # Add target load frame to dictionary
    #     target_frames[(load_x_frame, load_y_frame)] = [0, -magnitude, 0]
    


    return support_frames, target_frames, inventory, length

def set_multiple_cantilever_env_framegrid(
        frame_grid_size_x, 
        height_options = [1, 2, 3],
        length_options = [3, 4, 5],
        magnitude_options = [300, 400, 500],
        inventory_options = [(10,10), (10,5), (5,5), (8,3)], # each tuple should have element number of free frame type
        num_target_loads = 2,
        seed=None,
        fixed_hlm = None):
    """
    Used in cantileverenv_V2.py (agent playable setting)
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

    Create one support frame and multiple external loads with height & length (right is positive)

    Input:
        - frame_grid_size_x: Number of frames along the x-dimension of the frame grid.
        - height_options: List of possible heights for the external loads
        - length_options: List of possible lengths for the external loads
        - magnitude_options: List of possible magnitudes for the external loads
        - inventory_options: List of possible inventory options for the free frames
        - num_target_loads = 2, (tree like structure)
        - seed=None,
        - fixed_hlm = None list of (height, length, magnitude) for fixed boundary condition
    
    Output:
        - support_frames : list of tuples (x_frame, y_frame) 
        - targetload_frames : dictionary ((x_frame,y_frame) : [x_forcemag, y_forcemag, z_forcemag]) tuples (force is applied in the negative y direction).
        - inventory : dictionary of FrameStructureType Free frame type : count 
    """

    # If a seed is provided, set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # light_inv, med_inv = random.choice(inventory_options)

    # inventory = {
    #     FrameStructureType.FST_10_10 : light_inv, # -1 indicate no limits
    #     FrameStructureType.FST_20_20 : med_inv,
    #     # FrameStructureType.HEAVY_FREE_FRAME : *,
    # }

    # Get free frame types
    free_frame_types = FrameStructureType.get_free_frame_types() # list of FrameStructureType
    num_free_frame_types = len(free_frame_types)

    assert len(inventory_options[0]) == num_free_frame_types, \
        f"Inventory options must have {num_free_frame_types} elements for each FrameStructureType, got {len(inventory_options[0])}."
    
    # Randomly select inventory counts for each frame type
    selected_inventory = random.choice(inventory_options)
    
    # Create inventory dictionary from inventory options
    inventory = {
        frame_type: selected_inventory[i]
        for i, frame_type in enumerate(free_frame_types)
    }

    # Set pinned supports within the frame grid
    # x at center even coordinate of the frame grid
    # x_support_start_frame = (frame_grid_size_x // FRAME_SIZE - (FRAME_SIZE//2)) if (frame_grid_size_x // FRAME_SIZE - (FRAME_SIZE//2)) % FRAME_SIZE == 0 else (frame_grid_size_x // FRAME_SIZE - FRAME_SIZE)
    x_support_start_frame = int(frame_grid_size_x // FRAME_SIZE)
    y_support_frame = 0  # Base of the cantilever, first row of the frame grid

    # Create 1 support frame
    support_frames = [
            (x_support_start_frame, y_support_frame),  # Support frame at (x, y)
        ]
    
    # Choose num_target_loads of target load frames 
    max_cantilever_length = 0
    target_frames = dict()
    if fixed_hlm == None:
        for i in range(num_target_loads):
            # Choose random height, length, and load magnitude
            height = random.choice(height_options)
            length = random.choice(length_options)
            magnitude = random.choice(magnitude_options)

            # Set target load frame within the frame grid
            load_x_frame = x_support_start_frame + (-1)**i * length # right left alternate
            load_y_frame = y_support_frame + height
            # Target load in negative y direction (z assumed to be zero)
            target_frames[(load_x_frame, load_y_frame)] = [0, -magnitude, 0]
            if length > max_cantilever_length:
                max_cantilever_length = length

    else: # fixed boundary conditions 
        assert num_target_loads == len(fixed_hlm)
        for i, target in enumerate(fixed_hlm):
            height, length, magnitude = target

            # Set target load frame within the frame grid
            load_x_frame = x_support_start_frame + (-1)**i * length # right left alternate
            load_y_frame = y_support_frame + height
            # Target load in negative y direction (z assumed to be zero)
            target_frames[(load_x_frame, load_y_frame)] = [0, -magnitude, 0]
            if length > max_cantilever_length:
                max_cantilever_length = length


    return support_frames, target_frames, inventory, max_cantilever_length

def set_cantilever_bcs(
        bcs=None, # list of Boundary Condition Dictionaries 
        frame_grid_size_x= None,
        frame_grid_size_y= None,
        seed=None,
        ):
    """
    Used in cantileverenv_V2.py (agent playable setting)
    Return boundary condition information to create supports, target load frames and set inventory for the episode
    Input:
        list of boundary condition dictionaries where 
            Boundary Condition Dictionaries
                targets: List[Tuple[int, int, float]]
                supports: List[Tuple[int, int]]
                inventory: Tuple[int, ...]
    
    Output: 
        one set of boundary conditions from bc list 
            - support_frames : list of tuples (x_frame, y_frame) 
            - targetload_frames : dictionary ((x_frame, y_frame) : [x_forcemag, y_forcemag, z_forcemag]) tuples (force is applied in the negative y direction).
            - inventory : dictionary of FrameStructureType Free frame type : count 
            - max_cantilever_length: int, maximum length of the cantilever in frames
    """
    # Assert that framegrid size is provided
    if frame_grid_size_x is None or frame_grid_size_y is None:
        raise ValueError("Frame grid size must be provided. Please set frame_grid_size_x and frame_grid_size_y.")
    
    # If a seed is provided, set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Select a random boundary condition from the list
    if bcs is None or len(bcs) == 0:
        raise ValueError("Boundary conditions list is empty or None. Please provide a valid list of boundary conditions.")
    bc = random.choice(bcs)

    ## Set inventory dictionary
    # Get free frame types
    free_frame_types = FrameStructureType.get_free_frame_types() # list of FrameStructureType
    num_free_frame_types = len(free_frame_types)

    assert len(bc['inventory']) == num_free_frame_types, \
        f"Inventory options must have {num_free_frame_types} elements for each FrameStructureType, got {len(bc['inventory'])}."
    
    # Create inventory dictionary from inventory options
    inventory = {
        frame_type: bc['inventory'][i] # key is FrameStructureType, value is count
        for i, frame_type in enumerate(free_frame_types)
    }

    ## Set origin coordinates for support frames and target load frames
    origin_frame_x = int(frame_grid_size_x // FRAME_SIZE)
    origin_frame_y = 0  # Base of the cantilever, first row of the frame grid

    ## Set support frames
    # Validate and create support frames in a single loop
    support_frames = []
    for x, y in bc['supports']:
        if y < 0 or y >= frame_grid_size_y:
            raise ValueError(f"Support frame y-coordinate {y} is out of bounds. Must be between 0 and {frame_grid_size_y - 1}.")
        support_frames.append((origin_frame_x + x, origin_frame_y + y))

    ## Set target load frames
    # Create target load frames from the bc['targets'] list
    target_frames = dict()
    max_cantilever_length = max(abs(y) for _, y, _ in bc['targets']) if bc['targets'] else 0 # max abs(y) value from targets 

    for x, y, magnitude in bc['targets']:
        # Validate y bounds
        if y < 1 or y >= frame_grid_size_y:
            raise ValueError(f"Target frame y-coordinate {y} is out of bounds. Must be between 1 and {frame_grid_size_y - 1}.")
        
        # Set target load frame within the frame grid
        load_x_frame = origin_frame_x + x
        load_y_frame = origin_frame_y + y
        
        # Target load in negative y direction (z assumed to be zero)
        target_frames[(load_x_frame, load_y_frame)] = [0, -magnitude, 0]

    return support_frames, target_frames, inventory, max_cantilever_length