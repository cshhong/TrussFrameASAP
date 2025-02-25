import numpy as np

'''
Used in cantileverenv_v0.py
communicate (python) FEAGraph -> (julia) ASAP FEA -> (python) displacement 
** python list -> Julia Vector , python tuple -> Julia tuple
    ASAP expects Vectors!
'''
def solve_fea(jl, feagraph, frame_length_m):
    '''
    Input
        jl : Julia session
        feagraph : FEAGraph object 
    
    extracts and reformat info from FEAGraph to input into julia function create_and_solve_model (created in truss_analysis.jl)
        node_coordinates    : list of [x, y] int coordinates of each vector
        element_connections : list of [i, j] node index pairs 
        fixed_idx           : list of node_idx that are fixed
        loads               : list of (node_idx, (load_x, load_y, load_z))

    Output
        max_displacement: (float) The maximum nodal displacement of the truss model.
        failed_elements: list of element index pairs that failed
    '''
    node_coordinates = [list(v.coordinates)+[0.0] for v in list(feagraph.vertices.values())]
    # node_coordinates = [(float(x), float(y)) for x, y in (v.coordinates for v in list(feagraph.vertices.values()))]
    element_connections = feagraph.edges
    fixed_idx = [feagraph.vertices[support].id for support in feagraph.supports]
    loads = [list((v.id, v.load)) for v in list(feagraph.vertices.values())]
    # print(f'(solve fea) loads : {loads}')

    # model = jl.create_and_solve_model_julia(node_coordinates, element_connections, fixed_idx, loads, frame_length_m)
    # displacement = model.u

    displacement, axial_forces, P_cap_kN =  jl.create_and_solve_model_julia(node_coordinates, element_connections, fixed_idx, loads, frame_length_m)

    displacement = np.array(displacement).reshape(len(node_coordinates), 6) # reshape to 3D coordinates (Frame Model)
    translational_u = displacement[:, :3]  # Extract the first three translational DOFs (u_x, u_y, u_z)

    # edge indices only (outdated)
    # failed_elements_idx = [i for i, force in enumerate(axial_forces) if abs(force) > P_cap_kN]
    # failed_elements = [element_connections[i] for i in failed_elements_idx]

    # Get list of ((node_idx1, node_idx2), signed_force) pairs of failed elements
    failed_elements = [(element_connections[i], force) for i, force in enumerate(axial_forces) if abs(force) > P_cap_kN]

    # print("Elements that failed:", failed_elements)


    # increment = 20
    # element_forces = jl.get_element_forces(model, increment)
    # print(f" element forces : {element_forces}")

    return translational_u, failed_elements