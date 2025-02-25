'''
Functions to store and query data from hdf5 files.

HDF5 file format:
    Root/
    │
    ├── EnvRenderProperties/
    │   ├── figsize (1x2 array)             # Figure size for rendering [width, height]
    │   ├── board_size_x (scalar)           # Size of the board in the x direction
    │   ├── board_size_y (scalar)           # Size of the board in the y direction
    │   └── allowable_deflection (scalar)   # Maximum allowable deflection for the environment
    │
    ├── Episodes/
    │   ├── eps_{eps_idx}/                      # Each episode is stored under a unique identifier
    │   │   ├── FEAGraph/
    │   │   │   ├── vertices/
    │   │   │   │   ├── coordinates (Nx2 or Nx3 array)  # (x, y) or (x, y, z) for each vertex
    │   │   │   │   ├── ids (Nx1 array)                 # unique integer IDs for vertices
    │   │   │   │   ├── is_free (Nx1 boolean array)     # free or pinned status for each vertex
    │   │   │   │   └── loads (Nx3 array)               # load vectors [load_x, load_y, load_z] for each vertex
    │   │   │   │
    │   │   │   ├── supports (Mx2 array)                # coordinates of supported vertices
    │   │   │   ├── edges (Ex2 array)                   # pairs of vertex indices representing edges
    │   │   │   ├── maximal_edges/
    │   │   │   │   ├── horizontal/                     # lists of maximal edges in specific directions
    │   │   │   │   ├── vertical/
    │   │   │   │   ├── diagonal_LT_RB/
    │   │   │   │   └── diagonal_LB_RT/
    │   │   │   │
    │   │   │   ├── external_loads/
    │   │   │   │   ├── coordinates (Lx2 array)         # coordinates of load locations
    │   │   │   │   └── load_values (Lx3 array)         # load vectors [load_x, load_y, load_z] at each location
    │   │   │   │
    │   │   │   ├── displacement (Nx3 array)            # nodal displacements [disp_x, disp_y, disp_z]
    │   │   │   └── failed_elements (Fx2 array)         # ((node_idx1, node_idx2), force_mag) for failed elements
    │   │   │
    │   │   ├── frames/
    │   │   │   ├── type_shapes (Px1 array)             # integer codes representing `FrameShapeType` [0,4)
    │   │   │   ├── centroids (Px2 array)               # (x, y) coordinates of frame centroids on board
    │   │   │   ├── frame_positions (Px2 array)         # (x_frame, y_frame) cell positions in frame grid
    │   │   │   ├── type_structures (Px1 array)         # integer codes representing `FrameStructureType` [0,5)
    │   │   │   └── ids (Px1 array)                     # unique frame IDs
    │   │   │       
    │   │   ├── frame_grid/
    │   │   │   └──frame_grid (NxM array)              # Frame grid with FrameStructureType ids at each cell - used for shape embedding
    │      
    ├── Clustering/
    │   ├── umap_n_neighbors (scalar)           # Number of neighbors for UMAP embedding
    │   ├── dbscan_num_core (scalar)            # Minimum number of points to form a dense region in DBSCAN
    │   ├── dbscan_eps (scalar)                 # Epsilon parameter for DBSCAN clustering
    │   ├── num_clusters (scalar)               # Number of clusters (excluding noise) detected by DBSCAN
    │   ├── cluster_labels (Nx1 array)          # Cluster labels for each frame; -1 indicates noise
    │
    ├── UMAP/
    │   ├── umap2d (Nx2)           # umap 2D embedding for each frame

'''

import h5py
# from TrussFrameASAP.TrussFrameMechanics.trussframe import FrameShapeType, FrameStructureType, TrussFrameRL
# from  TrussFrameASAP.TrussFrameMechanics.vertex import Vertex
# from  TrussFrameASAP.TrussFrameMechanics.maximaledge import MaximalEdge
# from  TrussFrameASAP.TrussFrameMechanics.feagraph import FEAGraph

from TrussFrameMechanics.trussframe import FrameShapeType, FrameStructureType, TrussFrameRL
from  TrussFrameMechanics.vertex import Vertex
from  TrussFrameMechanics.maximaledge import MaximalEdge
from  TrussFrameMechanics.feagraph import FEAGraph
import numpy as np
import pandas as pd

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

def save_umap2d_hdf5(hdf5_filename, umap2d):
    '''
    Input:
        - hdf5_filename(str): Path to the HDF5 file
        - umap2d(np.array): UMAP 2D embedding for each frame
    Save the UMAP 2D embedding to the hdf5 file.
    '''
    # Open the HDF5 file in append mode
    with h5py.File(hdf5_filename, 'a') as hdf5_obj:
        # Create or overwrite the group
        umap_group = hdf5_obj.require_group("UMAP")
        # Save UMAP 2D embedding
        # umap_group.create_dataset("umap2d", data=umap2d)
        # Check if "umap2d" dataset exists and overwrite if necessary
        if "umap2d" in umap_group:
            del umap_group["umap2d"]  # Delete the existing dataset
        
        # Create new dataset for "umap2d"
        umap_group.create_dataset("umap2d", data=umap2d)

        print("'umap2d' data saved successfully.")

def load_umap2d_hdf5(hdf5_filename):
    '''
    Input:
        - hdf5_filename(str): Path to the HDF5 file
    Load the UMAP 2D embedding from the hdf5 file.
    '''
    with h5py.File(hdf5_filename, 'r') as f:
        umap2d = f["UMAP/umap2d"][:]
        print("UMAP 2D embedding loaded successfully.")
    return umap2d

def save_cluster_data(hdf5_filename, umap_n_neighbors, dbscan_num_core, dbscan_eps, num_clusters, cluster_labels):
    '''
    Input:
        - hdf5_filename(str): Path to the HDF5 file
        - umap_n_neighbors(int): Number of neighbors for UMAP embedding
        - dbscan_num_core(int): Minimum number of points to form a dense region in DBSCAN
        - dbscan_eps(float): Epsilon parameter for DBSCAN clustering
        - num_clusters(int): Number of clusters detected by DBSCAN
        - cluster_labels(np.array): Cluster labels for each frame
    After finetuning clustering, save the clustering data to the hdf5 file.
    '''
    # Open the HDF5 file in append mode
    with h5py.File(hdf5_filename, 'a') as hdf5_obj:
        # Create or overwrite the group
        clustering_group = hdf5_obj.require_group("Clustering")
        # Save clustering data
        # clustering_group.create_dataset("umap_n_neighbors", data=umap_n_neighbors)
        # clustering_group.create_dataset("dbscan_num_core", data=dbscan_num_core)
        # clustering_group.create_dataset("dbscan_eps", data=dbscan_eps)
        # clustering_group.create_dataset("num_clusters", data=num_clusters)
        # clustering_group.create_dataset("cluster_labels", data=cluster_labels)
        # List of dataset names to save
        datasets = {
            "umap_n_neighbors": umap_n_neighbors,
            "dbscan_num_core": dbscan_num_core,
            "dbscan_eps": dbscan_eps,
            "num_clusters": num_clusters,
            "cluster_labels": cluster_labels
        }
        
        # Iterate over datasets and overwrite if they exist
        for name, data in datasets.items():
            if name in clustering_group:
                del clustering_group[name]  # Delete the existing dataset
            clustering_group.create_dataset(name, data=data)  # Create new dataset

        print("'Clustering' data saved successfully.")

def load_cluster_labels_hdf5(hdf5_filename):
    '''
    Input:
        - hdf5_filename(str): Path to the HDF5 file
    Load the cluster labels from the hdf5 file.
    '''
    with h5py.File(hdf5_filename, 'r') as f:
        cluster_params_group = f["Clustering"]
    
        # Load and print cluster parameters
        umap_n_neighbors = cluster_params_group["umap_n_neighbors"][()]
        dbscan_num_core = cluster_params_group["dbscan_num_core"][()]
        dbscan_eps = cluster_params_group["dbscan_eps"][()]
        num_clusters = cluster_params_group["num_clusters"][()]
        cluster_labels = cluster_params_group["cluster_labels"][()]

        print("Cluster Parameters:")
        print(f"    UMAP n_neighbors: {umap_n_neighbors}")
        print(f"    DBSCAN num_core: {dbscan_num_core}")
        print(f"    DBSCAN eps: {dbscan_eps}")
        print(f"    Number of Clusters: {num_clusters}")
        # print(f"    Cluster Labels: {cluster_labels}")
        # ! Check if number of clusters and cluster size is reasonable ! # 
        # get cluster idx - number of instances in each cluster
        cluster_idx, cluster_counts = np.unique(cluster_labels, return_counts=True)
        # print cluster idx - number of instances in each cluster in inline for loop
        print(f'cluster {cluster_idx} \n {cluster_counts}')
        print("Cluster labels loaded successfully.")

    return cluster_labels


def save_env_render_properties(hdf5_obj, env):
    '''
    Input:
        - hdf5_obj(h5py.File): An open HDF5 file object.
        - env: TrussFrameRL environment object
    At start of rollout save render properties to the hdf5 file.
    used for render loaded
    '''
    # Create or overwrite the group
    render_group = hdf5_obj.require_group("EnvRenderProperties")
    
    # Save datasets
    render_group.create_dataset("figsize", data=env.unwrapped.figsize)
    render_group.create_dataset("board_size_x", data=env.unwrapped.board_size_x)
    render_group.create_dataset("board_size_y", data=env.unwrapped.board_size_y)
    render_group.create_dataset("allowable_deflection", data=env.unwrapped.allowable_deflection)
    render_group.create_dataset("frame_size", data=env.unwrapped.frame_size)

    print("'EnvRenderProperties' saved successfully.")

def save_episode_hdf5(hdf5_obj, term_eps_idx, curr_fea_graph, frames, curr_frame_grid):
    '''
    Save episode data to an HDF5 file.

    Parameters:
        - hdf5_obj (h5py.File): An open HDF5 file object.
        - term_eps_idx (int): The index of the terminated episode.
        - curr_fea_graph (FEAGraph): The FEAGraph object for the terminated episode.
        - frames (list of TrussFrameRL): List of TrussFrameRL objects for the terminated episode.
        - curr_frame_grid (np.array): The frame grid for the terminated episode.
    '''
    # Ensure the encompassing Episodes group exists
    if "Episodes" not in hdf5_obj:
        large_episodes_group = hdf5_obj.create_group("Episodes", track_order=True)
    else:
        large_episodes_group = hdf5_obj["Episodes"]

    # Create group for the specific episode
    episode_group = large_episodes_group.create_group(f'eps_{term_eps_idx}')

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

def load_episode_hdf5(hdf5_filename, eps_idx):
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
    episode_path = f'Episodes/eps_{eps_idx}'
    
    with h5py.File(hdf5_filename, 'r') as f:
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
        failed_elements = [((int(el[0]), int(el[1])), force_mag) for (el, force_mag) in f[f'{episode_path}/FEAGraph/failed_elements'][:]]
        
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


# def load_framegrids_hdf5(hdf5_filename):
#     """
#     From an HDF5 file, for all episodes, load the framegrid np.array.

#     Parameters:
#         - hdf5_filename: str, path to the HDF5 file
#         - eps_range: tuple (start, end), optional
#             Specifies the range of episodes to load. 
#             If None, load all episodes. Episodes are zero-indexed.
        
#     Returns:
#         - stacked frame grids: np.array (num_episodes, num_rows, num_cols)
#     """
    
#     with h5py.File(hdf5_filename, 'r') as f:
#         # Access the Episodes group
#         if "Episodes" not in f:
#             raise KeyError("The 'Episodes' group is not found in the HDF5 file.")

#         episodes_group = f["Episodes"]
#         episode_keys = list(episodes_group.keys())
#         num_episodes = len(episode_keys)

#         # Determine the shape of the frame grid from the first episode
#         first_episode_key = episode_keys[0]
#         frame_grid_shape = episodes_group[f'{first_episode_key}/frame_grid/frame_grid'].shape

#         # Initialize the stacked frame grids array
#         stacked_frame_grids = np.zeros((num_episodes, frame_grid_shape[0], frame_grid_shape[1]))

#         # Iterate through each episode and load the frame grid
#         for i, episode in enumerate(episode_keys):
#             # print(f'Loading frame grid for episode {episode}')
#             frame_grid = episodes_group[f'{episode}/frame_grid/frame_grid'][:]
#             stacked_frame_grids[i] = frame_grid
        
#     return stacked_frame_grids


def load_framegrids_hdf5(hdf5_filename, eps_range=None):
    """
    From an HDF5 file, load the frame grids for a specified range of episodes.

    Parameters:
        - hdf5_filename: str, path to the HDF5 file
        - eps_range: tuple (start, end), optional
            Specifies the range of episodes to load. 
            If None, load all episodes. Episodes are zero-indexed.

    Returns:
        - stacked frame grids: np.array (num_selected_episodes, num_rows, num_cols)
    """
    with h5py.File(hdf5_filename, 'r') as f:
        # Access the Episodes group
        if "Episodes" not in f:
            raise KeyError("The 'Episodes' group is not found in the HDF5 file.")

        episodes_group = f["Episodes"]
        episode_keys = list(episodes_group.keys())

        # Determine the range of episodes to load
        if eps_range is None:
            selected_episode_keys = episode_keys
        else:
            start, end = eps_range
            selected_episode_keys = episode_keys[start:end]

        num_selected_episodes = len(selected_episode_keys)

        # Determine the shape of the frame grid from the first selected episode
        first_episode_key = selected_episode_keys[0]
        frame_grid_shape = episodes_group[f'{first_episode_key}/frame_grid/frame_grid'].shape

        # Initialize the stacked frame grids array
        stacked_frame_grids = np.zeros((num_selected_episodes, frame_grid_shape[0], frame_grid_shape[1]))

        # Iterate through each selected episode and load the frame grid
        for i, episode in enumerate(selected_episode_keys):
            # print(f'Loading frame grid for episode {episode}')
            frame_grid = episodes_group[f'{episode}/frame_grid/frame_grid'][:]
            stacked_frame_grids[i] = frame_grid

    return stacked_frame_grids

def load_max_deflection_hdf5(hdf5_filename, eps_range=None):
    """
    From an HDF5 file, for each episode in the specified range, load the displacement values 
    and get the maximum displacement.

    Parameters:
        - hdf5_filename: str, path to the HDF5 file
        - eps_range: tuple (start, end), optional
            Specifies the range of episodes to load. If None, load all episodes.
            Episodes are zero-indexed.

    Returns:
        - max_displacements: np.array (num_selected_episodes,)
    """
    with h5py.File(hdf5_filename, 'r') as f:
        # Access the Episodes group
        if "Episodes" not in f:
            raise KeyError("The 'Episodes' group is not found in the HDF5 file.")

        episodes_group = f["Episodes"]
        episode_keys = list(episodes_group.keys())

        # Determine the range of episodes to load
        if eps_range is None:
            selected_episode_keys = episode_keys
        else:
            start, end = eps_range
            selected_episode_keys = episode_keys[start:end]

        num_selected_episodes = len(selected_episode_keys)

        # Initialize array to store maximum displacements
        max_displacements = np.zeros(num_selected_episodes)

        # Iterate through each selected episode and compute max displacement
        for i, episode in enumerate(selected_episode_keys):
            # print(f'Loading displacement data for episode {episode}')
            # Load displacement data (Nx3 array: [disp_x, disp_y, disp_z])
            displacement_xyz = episodes_group[f'{episode}/FEAGraph/displacement'][:]

            # Compute displacement magnitudes: sqrt(disp_x^2 + disp_y^2 + disp_z^2)
            displacement_magnitude = np.linalg.norm(displacement_xyz, axis=1)

            # Find the maximum displacement magnitude
            max_displacements[i] = np.max(displacement_magnitude)

    return max_displacements

# def load_max_deflection_hdf5(hdf5_filename):
#     """
#     From an HDF5 file, for each episode, load the displacement values and get the maximum displacement.

#     Parameters:
#         - filename: str, path to the HDF5 file

#     Returns:
#         - max_displacements: np.array (num_episodes,)
#     """
#     with h5py.File(hdf5_filename, 'r') as f:
#         # Access the Episodes group
#         if "Episodes" not in f:
#             raise KeyError("The 'Episodes' group is not found in the HDF5 file.")

#         episodes_group = f["Episodes"]
#         episode_keys = list(episodes_group.keys())
#         num_episodes = len(episode_keys)

#         # Initialize array to store maximum displacements
#         max_displacements = np.zeros(num_episodes)

#         # Iterate through each episode and compute max displacement
#         for i, episode in enumerate(episode_keys):
#             # print(f'Loading displacement data for episode {episode}')
#             # Load displacement data (Nx3 array: [disp_x, disp_y, disp_z])
#             displacement_xyz = episodes_group[f'{episode}/FEAGraph/displacement'][:]

#             # Compute displacement magnitudes: sqrt(disp_x^2 + disp_y^2 + disp_z^2)
#             displacement_magnitude = np.linalg.norm(displacement_xyz, axis=1)

#             # Find the maximum displacement magnitude
#             max_displacements[i] = np.max(displacement_magnitude)

#     return max_displacements
    
def load_env_render_properties(hdf5_filename):
    """
    Loads the EnvRenderProperties from an HDF5 file and returns them as a dictionary.

    Parameters:
        hdf5_file (h5py.File): Open HDF5 file object.

    Returns:
        dict: Dictionary containing the render properties:
            - figsize: List [width, height]
            - board_size_x: Scalar
            - board_size_y: Scalar
            - allowable_deflection: Scalar
            - frame_size: Scalar

    """

    # Access the EnvRenderProperties group
    with h5py.File(hdf5_filename, 'r') as f:
        # Extract datasets and return as a dictionary
        render_properties = {
            "figsize": f["EnvRenderProperties/figsize"][:].tolist(),  # Convert to Python list
            "board_size_x": f["EnvRenderProperties/board_size_x"][()],
            "board_size_y": f["EnvRenderProperties/board_size_y"][()],
            "allowable_deflection": f["EnvRenderProperties/allowable_deflection"][()],
            "frame_size": f["EnvRenderProperties/frame_size"][()],
        }

        return render_properties

def load_failed_elements_hdf5(hdf5_filename, eps_range=None):
    """
    From an HDF5 file, for specified episodes, load the failed_elements ((Fx2 array) pairs of node indices for failed elements).

    Parameters:
        - hdf5_filename: str, path to the HDF5 file
        - eps_range: tuple (start, end), optional
            Specifies the range of episodes to load. If None, load all episodes.
            Episodes are zero-indexed.

    Returns:
        - 3D jagged list (num_selected_episodes, (num_failed_elements, 2))
    """
    with h5py.File(hdf5_filename, 'r') as f:
        # Access the Episodes group
        if "Episodes" not in f:
            raise KeyError("The 'Episodes' group is not found in the HDF5 file.")

        episodes_group = f["Episodes"]
        episode_keys = list(episodes_group.keys())

        # Determine the range of episodes to load
        if eps_range is None:
            selected_episode_keys = episode_keys
        else:
            start, end = eps_range
            selected_episode_keys = episode_keys[start:end]

        all_failed_elements = []

        # Iterate through each selected episode and load the failed elements
        for episode in selected_episode_keys:
            # print(f'Loading failed elements for episode {episode}')
            failed_elems = episodes_group[f'{episode}/FEAGraph/failed_elements'][:]
            all_failed_elements.append(failed_elems.tolist())

    # Filter indices where failed elements exist
    indices_with_failed_elements = [i for i, failed in enumerate(all_failed_elements) if len(failed) > 0]

    assert len(all_failed_elements) == len(selected_episode_keys), \
        f"Failed elements not loaded correctly; there should be {len(selected_episode_keys)} lists"
    return all_failed_elements

# def load_failed_elements_hdf5(hdf5_filename):
#     """
#     From an HDF5 file, for all episodes, load the failed_elements ((Fx2 array) pairs of node indices for failed elements).

#     Parameters:
#         - filename: str, path to the HDF5 file
        
#     Returns:
#         - 3D jagged list (num_episodes, (num_failed_elements, 2))
#     """
#     with h5py.File(hdf5_filename, 'r') as f:
#         # Access the Episodes group
#         if "Episodes" not in f:
#             raise KeyError("The 'Episodes' group is not found in the HDF5 file.")

#         episodes_group = f["Episodes"]
#         episode_keys = list(episodes_group.keys())
#         num_episodes = len(episode_keys)
#         all_failed_elements = []
#         # Iterate through each episode and load the frame grid
#         for i, episode in enumerate(episode_keys):
#             # print(f'Loading failed elem for episode {episode}')
#             failed_elems = episodes_group[f'{episode}/FEAGraph/failed_elements'][:]
#             all_failed_elements.append(failed_elems.tolist())
#     indices_with_failed_elements = [i for i, failed in enumerate(all_failed_elements) if len(failed) > 0]

#     assert len(all_failed_elements) == num_episodes, "Failed elements not loaded correctly there should be {num_episodes} lists"
#     return all_failed_elements

def load_allowable_deflection_hdf5(hdf5_filename):
    """
    From an HDF5 file, load the allowable deflection (scalar) for the environment.

    Parameters:
        - filename: str, path to the HDF5 file
        
    Returns:
        - scalar
    """
    with h5py.File(hdf5_filename, 'r') as f:
        allowable_deflection = f["EnvRenderProperties/allowable_deflection"][()]
    return allowable_deflection

def init_csv_from_hdf5(hdf5_filename, csv_filename):
    """
    Create a CSV file with columns 'idx' and 'img_url'.
    Populate idx values with episode indices from the HDF5 file.
    'img_url' values are set to None.

    Parameters:
        hdf5_filename (str): Path to the HDF5 file.
        csv_filename (str): Path to the output CSV file.

    Returns:
        None

    Hdf5 file structure:
    Root/
    │
    ├── EnvRenderProperties/
    │   ├── figsize (1x2 array)             # Figure size for rendering [width, height]
    │   ├── board_size_x (scalar)           # Size of the board in the x direction
    │   ├── board_size_y (scalar)           # Size of the board in the y direction
    │   └── allowable_deflection (scalar)   # Maximum allowable deflection for the environment
    │
    ├── Episodes/
    │   ├── eps_{eps_idx}/                      # Each episode is stored under a unique identifier
    ...

    """
    # Open the HDF5 file
    with h5py.File(hdf5_filename, "r") as hdf5_file:
        # Ensure the file has the expected structure
        if "Episodes" not in hdf5_file:
            raise KeyError("HDF5 file does not contain 'Episodes' group.")
        
        # Extract episode indices from the HDF5 file
        episode_keys = list(hdf5_file["Episodes"].keys())
        episode_indices = [int(key.split("_")[1]) for key in episode_keys]

    # Create a DataFrame with 'idx' and 'img_url' columns
    data = {"idx": episode_indices, "img_url": [None] * len(episode_indices)}
    df = pd.DataFrame(data)

    # Save the DataFrame to the CSV file
    df.to_csv(csv_filename, index=False)
    print(f"CSV file created: {csv_filename}")