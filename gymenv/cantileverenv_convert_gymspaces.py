'''
Encode and Decode gym observation and action spaces for canilever-v0 environment
Because cleanRL only uses gym.spaces.Discrete (discrete) spaces
    - MultiDiscrete values with individual bounds are encoded into a single integer

'''

import numpy as np
import torch
import torch.nn.functional as F

class ObservationDownSamplingMapping:
    """
    Observation format:
        A tuple consisting of:
        1. `full_framegrid_occupancy` (array of size `(framegrid_size_x, framegrid_size_y)`):
            - Represents the occupancy status of each cell in the frame grid.
            - Each cell has bounds `[-1, len(inventory_array)]`, where:
                - `-1`: Represents an external force.
                - `0`: Represents an unoccupied cell.
                - `1`: Represents a support frame.
                - `2`: Represents a light free frame.
                - `3`: Represents a medium free frame.

        2. `current_inventory` (array of size `(len(inventory_array),)`):
            - Represents the current inventory level for each free frame type.
            - Each entry has bounds `[0, frame_type_inventory_cap]`.

    Dimensions:
        - `framegrid_size_x`: Number of cells in the x-dimension of the frame grid.
        - `framegrid_size_y`: Number of cells in the y-dimension of the frame grid.
        - `inventory_array`: A list where each element defines the maximum inventory capacity for a specific frame type.
    
    Down Sampling: (difference from ObservationBijectiveMapping)
        - The frame grid is downsampled to a smaller grid size with convolution and average pooling.
        - not bijective, no decoder

    Encoded Observation:
        np.array [encoded_framegrid_value, inventory*]
        - The downsampled frame grid is encoded into a single integer.
            - with a base of `len(inventory_array) + 2` (including -1 and 0).
        - The inventory is current inventory levels for each free frame type.

    """
    def __init__(self, framegrid_size_x, framegrid_size_y, inventory_array, kernel_size=2, stride=1):
        """
        Initialize the mapping with grid sizes and inventory bounds.
        :param framegrid_size_x: Number of columns in the frame grid.
        :param framegrid_size_y: Number of rows in the frame grid.
        :param inventory_array: A list of maximum inventory levels for each frame type.
        """
        self.framegrid_size_x = framegrid_size_x
        self.framegrid_size_y = framegrid_size_y
        self.inventory_array = inventory_array

        # Bounds for grid occupancy and inventory
        self.grid_bounds = (-1, len(inventory_array)+2)  # [-1, len(inventory_array)+2] for frame grid (persists for downsampled grid)
        self.grid_encoding_base = len(inventory_array) + 2  # Add 2 for -1 and 0 bounds
        self.inventory_encoding_base = [cap for cap in inventory_array]  # Inventory bounds

        # Downsampling parameters
        self.kernel_size = kernel_size
        self.stride = stride

        # Total space size for validation
        self.total_space_size = self._calculate_total_space_size()
        print(f"observation space size: {self.total_space_size} smaller than C long? 64 bit {self.total_space_size < 2*9223372036854775808} 32 bit {self.total_space_size < 2*2147483647}")
        self.encoded_reduced_framegrid_min, self.encoded_reduced_framegrid_max = self._calculate_encoded_grid_bounds()
        # print(f'encoded reduced framegrid bounds : {self.encoded_reduced_framegrid_min, self.encoded_reduced_framegrid_max}')

    def _calculate_encoded_grid_bounds(self):
        """
        Calculate the minimum and maximum bounds for the encoded downsampled grid values.
        Since each downsampled grid cell can take on [-1, len(inventory_array) + 2] values, and these
        are combined as digits in a positional number system (base = len(inventory_array) + 3),
        the maximum encoded value is (base^(num_cells)) - 1, assuming each cell is at its max value.

        :return: (min_encoded_value, max_encoded_value)
        """

        # Compute padding (used by average pooling)
        padding = self.kernel_size // 2

        # Compute reduced dimensions after downsampling
        def out_dim(in_size, k_size, stride, pad):
            # Formula for output dimension of convolution/pooling:
            # out = floor((in_size + 2*pad - k_size) / stride) + 1
            return (in_size + 2*pad - k_size) // stride + 1

        reduced_x = out_dim(self.framegrid_size_x, self.kernel_size, self.stride, padding)
        reduced_y = out_dim(self.framegrid_size_y, self.kernel_size, self.stride, padding)

        # Number of cells in the downsampled grid
        num_downsampled_cells = reduced_x * reduced_y

        # Base for encoding a single cell is (len(inventory_array) + 2)
        base = self.grid_encoding_base

        # Minimum encoded value (all cells at their minimum) is 0
        # because we offset the range by subtracting `grid_bounds[0]`.
        min_encoded_value = 0

        # Maximum encoded value (all cells at their maximum)
        # Each cell can be represented as a digit in base 'base'
        # so the maximum number for N digits is base^N - 1
        max_encoded_value = (base ** num_downsampled_cells) - 1

        return min_encoded_value, max_encoded_value
 
    
    def _downsample_obs_framegrid(self, frame_grid, kernel_size, stride):
        """
        Downsamples the frame grid using convolution with a specified kernel size and stride.
        :param frame_grid: np.array of size (framegrid_size_x, framegrid_size_y).
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride of the convolution.
        :return: np.array of the downsampled grid. still in the range [-1, len(inventory_array)]
        """
        # Convert the frame grid to a PyTorch tensor and add batch and channel dimensions
        frame_grid_tensor = torch.tensor(frame_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Apply padding of 0 around the periphery
        padding = kernel_size // 2  # Add padding to maintain context at edges
        padded_tensor = F.pad(frame_grid_tensor, (padding, padding, padding, padding), mode='constant', value=0)
        # Perform convolution with average pooling
        downsampled_tensor = F.avg_pool2d(padded_tensor, kernel_size=kernel_size, stride=stride)
        # Convert the result back to a NumPy array
        downsampled_grid = downsampled_tensor.squeeze().numpy()
        # print(f'Size of downsampled grid: {downsampled_grid.shape}')
        return downsampled_grid
    
    def encode(self, org_frame_grid, inventory):
        """
        Encodes the original frame grid and inventory into a single integer.
        Downsample frame grid and encode it with inventory array
        :param downsampled_frame_grid: np.array of size (new_x, new_y) after downsampling.
        :param inventory: np.array of size (len(inventory_array),).
        :return: Encoded integer representing the observation.
        """
        flat_reduced_grid = self._downsample_obs_framegrid(org_frame_grid, self.kernel_size, self.stride).flatten()
        encoded_framegrid_value = 0
        multiplier = 1

        # Encode the downsampled frame grid
        for value in flat_reduced_grid:
            if not (self.grid_bounds[0] <= value <= self.grid_bounds[1]):
                print(f"downsampled frame grid: {flat_reduced_grid} has grid value out of {self.grid_bounds}")
                print(f'original frame grid: {org_frame_grid}')
                raise ValueError(f"Grid value {value} is out of bounds {self.grid_bounds}")
            encoded_framegrid_value += (value - self.grid_bounds[0]) * multiplier
            multiplier *= self.grid_encoding_base

        encoded_array = np.concatenate(([encoded_framegrid_value], inventory))
        return encoded_array
        # # Encode the inventory levels
        # for idx, value in enumerate(inventory):
        #     if not (0 <= value <= self.inventory_encoding_base[idx]):
        #         raise ValueError(f"Inventory value {value} at index {idx} is out of bounds (0, {self.inventory_encoding_base[idx]})")
        #     encoded_value += value * multiplier
        #     multiplier *= self.inventory_encoding_base[idx] + 1
        
        # discrete_encoded_value = int(encoded_value)

        # return discrete_encoded_value

    def _calculate_total_space_size(self):
        """
        Calculate the total size of the encoded observation space with downsampling.
        :return: Integer size of the entire observation space.
        """
        downsampled_x = (self.framegrid_size_x + self.kernel_size - 1) // self.kernel_size
        downsampled_y = (self.framegrid_size_y + self.kernel_size - 1) // self.kernel_size

        grid_space_size = (self.grid_encoding_base) ** (downsampled_x * downsampled_y)
        inventory_space_size = np.prod([cap + 1 for cap in self.inventory_array])
        return grid_space_size * inventory_space_size

    def validate_observation(self, frame_grid, inventory):
        """
        Validates the downsampled frame grid and inventory against their respective bounds.
        :param frame_grid: 2D array of size (framegrid_size_x, framegrid_size_y).
        :param inventory: 1D array of size len(inventory_array).
        :raises ValueError: If frame grid or inventory is out of bounds.
        """
        if not np.all((self.grid_bounds[0] <= frame_grid) & (frame_grid <= self.grid_bounds[1])):
            raise ValueError(f"Frame grid values are out of bounds: {self.grid_bounds}")

        for idx, val in enumerate(inventory):
            if not (0 <= val <= self.inventory_array[idx]):
                raise ValueError(f"Inventory index {idx} value {val} is out of bounds (0, {self.inventory_array[idx]})")



class ObservationBijectiveMapping:
    """
    Observation format:
        A tuple consisting of:
        1. `full_framegrid_occupancy` (array of size `(framegrid_size_x, framegrid_size_y)`):
            - Represents the occupancy status of each cell in the frame grid.
            - Each cell has bounds `[-1, len(inventory_array)]`, where:
                - `-1`: Represents an external force.
                - `0`: Represents an unoccupied cell.
                - `1`: Represents a support frame.
                - `2`: Represents a light free frame.
                - `3`: Represents a medium free frame.

        2. `current_inventory` (array of size `(len(inventory_array),)`):
            - Represents the current inventory level for each free frame type.
            - Each entry has bounds `[0, frame_type_inventory_cap]`.

    Dimensions:
        - `framegrid_size_x`: Number of cells in the x-dimension of the frame grid.
        - `framegrid_size_y`: Number of cells in the y-dimension of the frame grid.
        - `inventory_array`: A list where each element defines the maximum inventory capacity for a specific frame type.
    """

    def __init__(self, framegrid_size_x, framegrid_size_y, inventory_array):
        """
        Initialize the mapping with grid sizes and inventory bounds.
        :param framegrid_size_x: Number of columns in the frame grid.
        :param framegrid_size_y: Number of rows in the frame grid.
        :param inventory_array: A list of maximum inventory levels for each frame type.
        """
        self.framegrid_size_x = framegrid_size_x
        self.framegrid_size_y = framegrid_size_y
        self.inventory_array = inventory_array

        # Bounds for grid occupancy and inventory
        self.grid_bounds = (-1, len(inventory_array)+2)  # [-1, len(inventory_array)+2] for frame grid
        self.grid_encoding_base = len(inventory_array) + 2  # Add 2 for -1 and 0 bounds
        self.inventory_encoding_base = [cap for cap in inventory_array]  # Inventory bounds

        # Total space size for validation
        self.total_space_size = self._calculate_total_space_size()
        # Total space size for validation
        print(f"observation space size: {self.total_space_size} smaller than C long? {self.total_space_size < 2*9223372036854775808}")

  
    
    def _calculate_total_space_size(self):
        """
        Calculate the total size of the encoded observation space.
        :return: Integer size of the entire observation space.
        """
        grid_size = self.framegrid_size_x * self.framegrid_size_y
        grid_space = self.grid_encoding_base ** grid_size
        inventory_space = np.prod(self.inventory_encoding_base)
        return grid_space * inventory_space
    


    def encode(self, frame_grid, inventory):
        """
        Encodes the frame grid and inventory into a single integer.
        Input
            frame_grid : np.array of size (framegrid_size_x, framegrid_size_y).
            inventory : np.array of size  (inventory_array, ).
        """
        # Validate inputs
        self.validate_observation(frame_grid, inventory)

        # Flatten the frame grid and encode it
        flattened_grid = frame_grid.flatten()
        grid_encoded = 0
        multiplier = 1
        for value in flattened_grid:
            grid_encoded += (value + 1) * multiplier  # Shift value by +1 to handle -1
            multiplier *= self.grid_encoding_base

        # Encode inventory
        inventory_encoded = 0
        multiplier = 1
        for value, bound in zip(inventory, self.inventory_encoding_base):
            inventory_encoded += value * multiplier
            multiplier *= bound

        # Combine grid and inventory encodings
        total_grid_space = self.grid_encoding_base ** len(flattened_grid)  # Total space for grid
        encoded_value = grid_encoded + inventory_encoded * total_grid_space
        return encoded_value

    def decode(self, encoded_value):
        """
        Decodes a single integer into the frame grid and inventory.
        """
        total_grid_cells = self.framegrid_size_x * self.framegrid_size_y

        # Decode inventory
        total_grid_space = self.grid_encoding_base ** total_grid_cells  # Total space for grid
        inventory_encoded = encoded_value // total_grid_space
        encoded_value %= total_grid_space

        # inventory = []
        # for bound in reversed(self.inventory_encoding_base):
        #     inventory.append(inventory_encoded % bound)
        #     inventory_encoded //= bound
        # inventory.reverse()
        inventory = []
        for bound in self.inventory_encoding_base:
            inventory.append(inventory_encoded % bound)
            inventory_encoded //= bound

        # Decode frame grid
        grid_encoded = encoded_value
        flattened_grid = []
        for _ in range(total_grid_cells):
            flattened_grid.append((grid_encoded % self.grid_encoding_base) - 1)  # Undo +1 shift
            grid_encoded //= self.grid_encoding_base
        frame_grid = np.array(flattened_grid).reshape(self.framegrid_size_x, self.framegrid_size_y)

        return frame_grid, np.array(inventory)

    def validate_observation(self, frame_grid, inventory):
        """
        Validates the frame grid and inventory against their respective bounds.
        :param frame_grid: 2D array of size (framegrid_size_x, framegrid_size_y).
        :param inventory: 1D array of size len(inventory_array).
        :raises ValueError: If frame grid or inventory is out of bounds.
        """
        if frame_grid.shape != (self.framegrid_size_x, self.framegrid_size_y):
            raise ValueError(f"Frame grid must have shape ({self.framegrid_size_x}, {self.framegrid_size_y}).")
        if not ((frame_grid >= self.grid_bounds[0]).all() and (frame_grid <= self.grid_bounds[1]).all()):
            raise ValueError(f"Frame grid values must be in range {self.grid_bounds}.")
        if len(inventory) != len(self.inventory_array):
            raise ValueError(f"Inventory must have size {len(self.inventory_array)}.")
        if not all(0 <= inv <= cap for inv, cap in zip(inventory, self.inventory_array)):
            raise ValueError("Inventory values must be within their respective capacity bounds.")


class ActionBijectiveMapping:
    """
    Action Format:
        A tuple consisting of:
        1. `end_bool` (int): A boolean value indicating whether to end the action sequence.
            - Bounds: [0, 1]
            - 0: Continue the action sequence.
            - 1: End the action sequence.

        2. `frame_type` (int): The type of frame being placed.
            - Bounds: [freeframe_min, freeframe_max]

        3. `framegrid_coord_x` (int): The x-coordinate of the frame's position on the grid.
            - Bounds: [0, self.frame_grid_size_x]

        4. `framegrid_coord_y` (int): The y-coordinate of the frame's position on the grid.
            - Bounds: [0, self.frame_grid_size_y]
    """
    def __init__(self, frame_grid_size_x, frame_grid_size_y, freeframe_idx_min, freeframe_idx_max):
        """
        Initialize the mapping with action bounds.
        :param frame_grid_size_x: Maximum value for frame_x.
        :param frame_grid_size_y: Maximum value for frame_y.
        :param freeframe_idx_min: Minimum value for freeframe_type.
        :param freeframe_idx_max: Maximum value for freeframe_type.
        """
        self.action_bounds = [
            2,  # Bound for end_bool (0 or 1)
            freeframe_idx_max - freeframe_idx_min + 1,  # Bound for freeframe_type
            frame_grid_size_x,  # Bound for frame_x
            frame_grid_size_y,  # Bound for frame_y
        ]
        self.freeframe_idx_min = freeframe_idx_min
        self.total_space_size = self._calculate_total_space_size()

    def _calculate_total_space_size(self):
        """
        Calculate the total size of the encoded action space.
        """
        total_size = 1
        for bound in self.action_bounds:
            total_size *= bound
        return total_size

    def encode(self, action):
        """
        Encodes an action into a single integer.
        :param action: [end_bool, freeframe_type, frame_x, frame_y]
            * make sure that freeframe_type bounds start at 0 instead of freeframe_idx_min 
        :return: Encoded integer
        """
        action_cat = ['end_bool', 'freeframe_type', 'frame_x', 'frame_y']
        encoded_value = 0
        multiplier = 1
        # for value, bound in zip(action, self.action_bounds):
        for idx, (value, bound) in enumerate(zip(action, self.action_bounds)):
            if not (0 <= value < bound):
                # TODO get action category based on index from [end_bool, freeframe_type, frame_x, frame_y]
                raise ValueError(f"Value {value} is out of bounds for its bound {bound - 1} for action {action_cat[idx]}.")
            encoded_value += value * multiplier
            multiplier *= bound
        return encoded_value

    def decode(self, encoded_value):
        """
        Decodes a single integer into an action.
        :param encoded_value: Encoded integer
        :return list of integer [end_bool, freeframe_type, frame_x, frame_y]
        """
        # print(f'encoded value: {encoded_value}')
        if not (0 <= encoded_value < self.total_space_size):
            raise ValueError(f"Encoded value {encoded_value} is out of bounds for the total space size {self.total_space_size}.")
        action = []
        for bound in self.action_bounds:
            action.append(encoded_value % bound)
            encoded_value //= bound

        # NOTFIX without inplace modification of encoded values, vector values become redundant... why?

        # Ensure all elements are integers and not arrays or tensors
        action = [int(a) if not isinstance(a, (np.ndarray, torch.Tensor)) else int(a.item()) for a in action]

        return action