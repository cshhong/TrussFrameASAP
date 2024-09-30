import juliacall
import numpy as np
# from juliacall import Main as jl
'''
Wrapper to communicate (python) Graph -> (julia) ASAP FEA -> (python) displacement 

_extract_graph_data : extract from graph information needed for finite element analysis on structure using ASAP
solve_truss_from_graph : calls julia, performs FEA on structure within julia session, returns structure displacement to python
'''


def _extract_graph_data(graph):
    '''
    Input
        Graph object with vertices, edges, maximal_edges, loads
    Output
    ------
    node_coordinates : list of tuple float?(x, y, z)
    element_connections : list of tuple int(vertex1, vertex2)
    '''
    node_coordinates = []
    element_connections = []

    # Extract node coordinates
    # Sort vertices by Vertex.id and extract coordinates in sorted order
    sorted_vertices = sorted(graph.vertices.items(), key=lambda item: item[1].id)
    for coord, vertex in sorted_vertices:
        # Julia interprets Python lists as Julia vectors
        # Ensure the coordinates have 3 components (x, y, z)
        coordinates_3d = list(vertex.coordinates) + [0.0] if len(vertex.coordinates) == 2 else list(vertex.coordinates)
        node_coordinates.append(coordinates_3d)

    # Extract element connections (edges are already tuples of vertex indices)
    element_connections = graph.edges

    support_idx = list(graph.supports.keys()) # list of node idx that are supports / pinned as opposed to free
    
    return node_coordinates, element_connections, support_idx

    

def solve_truss_from_graph(jl, graph, loads):
    """
    Extracts node coordinates and element connections from a Graph object,
    creates a TrussModel in Julia using the extracted data, solves it,
    and returns the maximum displacement.
    
    Input:
    -----------
    graph : Graph
        The Graph object containing vertices and edges.
    
    Output:
    --------
    max_displacement : float
        The maximum nodal displacement of the truss model.
    """
    # Step 1: Extract information from the Graph object
    node_coords, element_conns, support_idx = _extract_graph_data(graph) # is format PyList{Any} in Julia

    # Step 2: Create and solve the Truss Model in Julia named 'model' using the extracted data
    # truss_model = jl.create_and_solve_truss_model_julia(node_coords, element_conns, loads)
    model = jl.create_and_solve_model_julia(node_coords, element_conns, support_idx, loads)

    # Extract the displacement from the solved model
    # displacement = truss_model.u # list of flattened displacement
    displacement = model.u # list of flattened displacement
    # print(f'displacement : {displacement}')
    # print(f'node_coords : {node_coords}')
    # translational_u = np.array(displacement).reshape(len(node_coords), 3) # reshape to 3D coordinates (Truss)
    displacement = np.array(displacement).reshape(len(node_coords), 6) # reshape to 3D coordinates (Frame Model)
    translational_u = displacement[:, :3]  # Extract the first three translational DOFs (u_x, u_y, u_z)

    print(f' displacement  : \n {translational_u}')


    return translational_u

