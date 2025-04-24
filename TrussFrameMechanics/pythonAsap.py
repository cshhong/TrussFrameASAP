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
    node_coordinates = [list(v.coordinates)+[0.0] for v in list(feagraph.vertices.values())] # add z coordinate
    fixed_idx = [feagraph.vertices[support].id for support in feagraph.supports]
    loads = [list((v.id, v.load)) for v in list(feagraph.vertices.values())]

    # Convert feagraph.edge_type_dict to Vector{Tuple{Int, Vector{Float64})
    # feagraph.edge_type_dict is dictionary where key : edge type int (weakest -> strongest), value : (outer diameter, inner wall thickness ratio) 
    element_types = []
    for idx, section in feagraph.edge_type_dict.items():
        element_types.append((idx, section[0], section[1])) # (edge_type, outer_diameter, inner_wall_thickness_ratio)
    
    element_id_type = feagraph.get_element_list_with_type() # list of v1_idx, v2_idx, element_type_idx)
    displacement, axial_forces, P_cap_kN =  jl.create_and_solve_model_julia(node_coordinates,     
                                                                             element_id_type, 
                                                                             fixed_idx, 
                                                                             loads,
                                                                             frame_length_m,
                                                                             element_types,
                                                                             )

    displacement = np.array(displacement).reshape(len(node_coordinates), 6) # reshape to 3D coordinates (Frame Model)
    translational_u = displacement[:, :3]  # Extract the first three translational DOFs (u_x, u_y, u_z)
    utilization = np.array(axial_forces) / np.array(P_cap_kN)  # Calculate utilization as a ratio of axial forces to capacity

    # Get list of ((node_idx1, node_idx2), signed_force) pairs of failed elements
    failed_elements = [((element_id_type[i][0], element_id_type[i][1]), force) for i, force in enumerate(axial_forces) if abs(force) > P_cap_kN[i]]
    # print("Elements that failed:", failed_elements)

    # increment = 20
    # element_forces = jl.get_element_forces(model, increment)
    # print(f" element forces : {element_forces}")
    return translational_u, failed_elements, utilization