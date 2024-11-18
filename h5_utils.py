'''
Functions to store and query data from hdf5 files.
HDF5 file format:
    Root/
    │
    ├── Episode_{eps_idx}/                      # Each episode is stored under a unique identifier
    │   ├── FEAGraph/
    │   │   ├── vertices/
    │   │   │   ├── coordinates (Nx2 or Nx3 array)  # (x, y) or (x, y, z) for each vertex
    │   │   │   ├── ids (Nx1 array)                 # unique integer IDs for vertices
    │   │   │   ├── is_free (Nx1 boolean array)     # free or pinned status for each vertex
    │   │   │   └── loads (Nx3 array)               # load vectors [load_x, load_y, load_z] for each vertex
    │   │   │
    │   │   ├── supports (Mx2 array)                # coordinates of supported vertices
    │   │   ├── edges (Ex2 array)                   # pairs of vertex indices representing edges
    │   │   ├── maximal_edges/
    │   │   │   ├── horizontal/                     # lists of maximal edges in specific directions
    │   │   │   ├── vertical/
    │   │   │   ├── diagonal_LT_RB/
    │   │   │   └── diagonal_LB_RT/
    │   │   │
    │   │   ├── external_loads/
    │   │   │   ├── coordinates (Lx2 array)         # coordinates of load locations
    │   │   │   └── load_values (Lx3 array)         # load vectors [load_x, load_y, load_z] at each location
    │   │   │
    │   │   ├── displacement (Nx3 array)            # nodal displacements [disp_x, disp_y, disp_z]
    │   │   └── failed_elements (Fx2 array)         # pairs of node indices for failed elements
    │   │
    │   ├── frames/
    │   │   ├── type_shapes (Px1 array)             # integer codes representing `FrameShapeType` [0,4)
    │   │   ├── centroids (Px2 array)               # (x, y) coordinates of frame centroids on board
    │   │   ├── frame_positions (Px2 array)         # (x_frame, y_frame) cell positions in frame grid
    │   │   ├── type_structures (Px1 array)         # integer codes representing `FrameStructureType` [0,5)
    │   │   └── ids (Px1 array)                     # unique frame IDs
    │   │       
    │   ├── frame_grid/
    │   │   ├── frame_grid (NxM array)              # Frame grid with FrameStructureType ids at each cell - used for shape embedding 

'''
import h5py
from TrussFrameMechanics.trussframe import FrameShapeType, FrameStructureType, TrussFrameRL
from  TrussFrameMechanics.vertex import Vertex
from  TrussFrameMechanics.maximaledge import MaximalEdge
from  TrussFrameMechanics.feagraph import FEAGraph
import numpy as np

def print_hdf5_structure(filename):
    with h5py.File(filename, 'r') as h5f:
        def print_attrs(name, obj):
            indent = '    ' * name.count('/')
            print(f"{indent}{name}:")
            if isinstance(obj, h5py.Group):
                print(f"{indent}  Group")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}  Dataset - shape: {obj.shape}, dtype: {obj.dtype}")
            for key, val in obj.attrs.items():
                print(f"{indent}  Attribute - {key}: {val}")
        h5f.visititems(print_attrs)

def save_episode_hdf5(h5f, term_eps_idx, curr_fea_graph, frames, curr_frame_grid):
    '''
    Save episode data to an HDF5 file.

    Parameters:
        - h5f (h5py.File): An open HDF5 file object.
        - term_eps_idx (int): The index of the terminated episode.
        - curr_fea_graph (FEAGraph): The FEAGraph object for the terminated episode.
        - frames (list of TrussFrameRL): List of TrussFrameRL objects for the terminated episode.
        - curr_frame_grid (np.array): The frame grid for the terminated episode.
    '''
    # Create root group for the specific episode
    episode_group = h5f.create_group(f'Episode_{term_eps_idx}')

    # Save FEAGraph data
    fea_graph_group = episode_group.create_group('FEAGraph')
    
    # Save vertices
    vertices_group = fea_graph_group.create_group('vertices')
    vertices_group.create_dataset('coordinates', data=[v.coordinates for v in curr_fea_graph.vertices.values()])
    vertices_group.create_dataset('ids', data=[v.id for v in curr_fea_graph.vertices.values()])
    vertices_group.create_dataset('is_free', data=[v.is_free for v in curr_fea_graph.vertices.values()])
    vertices_group.create_dataset('loads', data=[v.load for v in curr_fea_graph.vertices.values()])

    # Save supports
    fea_graph_group.create_dataset('supports', data=curr_fea_graph.supports)

    # Save edges
    fea_graph_group.create_dataset('edges', data=curr_fea_graph.edges)

    # Save maximal edges
    # maximal_edges_group = fea_graph_group.create_group('maximal_edges')
    # for direction, edges in curr_fea_graph.maximal_edges.items():
    #     maximal_edges_group.create_dataset(direction, data=edges)

    # Save external loads
    external_loads_group = fea_graph_group.create_group('external_loads')
    external_loads_group.create_dataset('coordinates', data=list(curr_fea_graph.external_loads.keys()))
    external_loads_group.create_dataset('load_values', data=list(curr_fea_graph.external_loads.values()))

    # Save displacement and failed elements
    fea_graph_group.create_dataset('displacement', data=curr_fea_graph.displacement)
    fea_graph_group.create_dataset('failed_elements', data=curr_fea_graph.failed_elements)

    # Save frames data
    frames_group = episode_group.create_group('frames')
    frames_group.create_dataset('type_shapes', data=[frame.type_shape.value for frame in frames])
    frames_group.create_dataset('centroids', data=[[frame.x, frame.y] for frame in frames])
    frames_group.create_dataset('frame_positions', data=[[frame.x_frame, frame.y_frame] for frame in frames])
    frames_group.create_dataset('type_structures', data=[frame.type_structure.idx for frame in frames])
    frames_group.create_dataset('ids', data=[frame.id for frame in frames])

    # Save frame grid data
    frame_grid_group = episode_group.create_group('frame_grid')
    frame_grid_group.create_dataset('frame_grid', data=curr_frame_grid)



def load_episode_hdf5(filename, eps_idx):
    """
    Load one episode of data from an HDF5 file. 
    Convert to FEAGraph object, and list of frames with TrussFrameRL objects.
    Parameters:
        - filename: str, path to the HDF5 file
        - eps_idx: int, the episode index to load
        
    Returns:
        - fea_graph: FEAGraph object
        - frames: list of TrussFrameRL objects
        - frame_grid : np.array of frame grid
        
    """
    episode_path = f'Episode_{eps_idx}'
    
    with h5py.File(filename, 'r') as f:
        # Load FEAGraph data
        vertices = {
            tuple(coord): Vertex(
                coordinates=tuple(coord),
                id=int(id_),
                is_free=bool(free),
                load=list(load)
            )
            for coord, id_, free, load in zip(
                f[f'{episode_path}/FEAGraph/vertices/coordinates'][:],
                f[f'{episode_path}/FEAGraph/vertices/ids'][:],
                f[f'{episode_path}/FEAGraph/vertices/is_free'][:],
                f[f'{episode_path}/FEAGraph/vertices/loads'][:]
            )
        }
        
        supports = [tuple(coord) for coord in f[f'{episode_path}/FEAGraph/supports'][:]]
        edges = [(int(edge[0]), int(edge[1])) for edge in f[f'{episode_path}/FEAGraph/edges'][:]]
        
        maximal_edges = {
            "horizontal": [MaximalEdge(...)]  # Add necessary loading for MaximalEdge objects
            # Add data loading for other directions
        }
        
        external_loads = {
            tuple(coord): list(load)
            for coord, load in zip(
                f[f'{episode_path}/FEAGraph/external_loads/coordinates'][:],
                f[f'{episode_path}/FEAGraph/external_loads/load_values'][:]
            )
        }
        
        displacement = f[f'{episode_path}/FEAGraph/displacement'][:]
        failed_elements = [(int(el[0]), int(el[1])) for el in f[f'{episode_path}/FEAGraph/failed_elements'][:]]
        
        # Construct FEAGraph
        fea_graph = FEAGraph(
            vertices=vertices,
            supports=supports,
            edges=edges,
            maximal_edges=maximal_edges,
            external_loads=external_loads,
            displacement=displacement,
            failed_elements=failed_elements
        )
        
        # Load frames
        frames = [
            TrussFrameRL(
                type_shape=FrameShapeType.get_frameshapetype_from_value(int(shape)),
                pos=(centroid[0], centroid[1]),
                type_structure=FrameStructureType.get_framestructuretype_from_idx(int(structure))
            )
            for shape, centroid, structure in zip(
                f[f'{episode_path}/frames/type_shapes'][:],
                f[f'{episode_path}/frames/centroids'][:],
                f[f'{episode_path}/frames/type_structures'][:]
            )
        ]

        # Load frame_grid
        frame_grid = f[f'{episode_path}/frame_grid/frame_grid'][:]
        
    return fea_graph, frames, frame_grid


def load_framegrids_hdf5(filename):
    """
    From an HDF5 file, for all episodes, load the framegrid np.array 
    Convert to FEAGraph object, and list of frames with TrussFrameRL objects.
    Parameters:
        - filename: str, path to the HDF5 file
        
    Returns:
        - stacked frame grids : np.array (num_episodes, num_rows, num_cols)
    """
    
    with h5py.File(filename, 'r') as f:
        num_episodes = len(f.keys())
        first_episode = list(f.keys())[0]
        frame_grid_shape = f[f'{first_episode}/frame_grid/frame_grid'].shape
        stacked_frame_grids = np.zeros((num_episodes, frame_grid_shape[0], frame_grid_shape[1]))
        
        for i, episode in enumerate(f.keys()):
            frame_grid = f[f'{episode}/frame_grid/frame_grid'][:]
            stacked_frame_grids[i] = frame_grid
        
    return stacked_frame_grids

