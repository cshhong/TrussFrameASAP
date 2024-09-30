import random

def set_cantilever_env(board_size_x, square_size):
    """
    Set up the cantilever environment with parametric boundary conditions.

    Input:
    - square_size: Size of each square in the grid

    Output:
    - pinned_supports: Dictionary where key is the node idx i, and value is (x, y) coordinates representing the pinned supports
                       (coordinates are on a square_size grid, spaced 2 units apart horizontally)
    - target_load: Dictionary where key is the (x, y) grid coordinate and value is the [x, y, z] force magnitude
                   (coordinates are on the square_size grid, force magnitude is applied in negative y direction)
    """

    # Sample boundary conditions
    num_supports_options = [1,2]
    height_options = [1, 2]
    length_options = [2, 3, 4]
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