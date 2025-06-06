"""
Julia functions that use ASAP to run finite element analysis
ASAP Common fixity types
const fixDict = Dict(:fixed => [false, false, false, false, false, false],
    :free => [true, true, true, true, true, true],
    :xfixed => [false, true, true, true, true, true],
    :yfixed => [true, false, true, true, true, true],
    :zfixed => [true, true, false, true, true, true],
    :xfree => [true, false, false, false, false, false],
    :yfree => [false, true, false, false, false, false],
    :zfree => [false, false, true, false, false, false],
    :pinned => [false, false, false, true, true, true])
"""

# truss_analysis.jl
module TrussAnalysis

# export create_and_solve_truss_model_julia
export create_and_solve_model_julia
export get_element_forces

    # using AsapToolkit
    using Asap
    
    """
    create_and_solve_truss_model_julia(node_coords, element_conns, loads)

    Creates and solves a truss model using provided node coordinates, element connections, and applied loads. 
    The function generates a `TrussModel` in Julia, solves it, and returns the solved model.

    Inputs:
    - `node_coords::Vector{Pylist{Any}}`: 
        A vector of vectors representing the 2D or 3D coordinates of each node. 
        Each node coordinate is a vector of the form `[x, y]` or `[x, y, z]`.
        
    - `element_conns::Vector{Pylist{Any}}`: 
        A vector of vectors where each inner vector represents an element's connection between two nodes, given by their indices. 
        Each connection is of the form `[i, j]`, where `i` and `j` are the node indices (1-based).

    - `loads::Vector{Tuple{Int, Vector{Float64}}}`: 
        A vector of tuples where each tuple represents a load applied to a node. 
        The first element of the tuple is the node index (1-based), and the second element is a vector representing the load in each direction, e.g., `[Fx, Fy, Fz]`.

    Outputs
    - `model::TrussModel`: 
        The solved `TrussModel` containing the nodes, elements, and displacements after solving.
        
    The model can be used to query results such as nodal displacements or reactions.
    """
    function create_and_solve_truss_model_julia(node_coords, element_conns, loads)
        
        # Define Nodes as fixed truss nodes (rotationally free displacement fixed) (two lines are the same)
        # nodes = TrussNode[TrussNode(Vector{Float64}(coord), :fixed) for coord in node_coords]
        # nodes = TrussNode[TrussNode(Vector{Float64}(coord), [false, false, false]) for coord in node_coords]
        # Define nodes with the first two nodes fixed and the rest pinned
        nodes = [TrussNode(Vector{Float64}(coord), i < 3 ? :fixed : [true, true, false]) for (i, coord) in enumerate(node_coords)]
        # println("Nodes: ", nodes)

        # Define material properties (assuming constant values)
        E = 70.0  # kN/m^2 Example Elastic modulus
        A = 4e3 / 1e6  # m^2 Example cross-sectional area
        sec = TrussSection(A, E)

        # Create elements using the connections
        # elements = TrussElement[TrussElement(nodes[conn[1]], nodes[conn[2]], sec) for conn in element_conns]
        elements = TrussElement[TrussElement(nodes[conn[1]], nodes[conn[2]], sec) for conn in element_conns]
        # println("element : ", elements[1] )

        # Create type NodeForce loads
        loads = NodeForce[NodeForce(nodes[load[1]], Vector{Float64}(load[2])) for load in loads]

        # Create the truss model
        model = TrussModel(nodes, elements, loads)

        # Solve the truss model
        planarize!(model)
        solve!(model)

        # DEBUG print displacement model.u
        # println(model.u)


        # println("fixed DOFs: ", model.fixedDOFs)
        # reactions = model.reactions[model.fixedDOFs]
        # reactions2d = reactions[reactions .!= 0]

        # println("Reactions: ", reactions2d)

        # ds = displacements(model, 2)
        # println("displacements within Julia", ds)

        # element = Element(n1, n2, section)
        # elements = [element]

        # curr_elem = elements[1]
        # element = Element(curr_elem.section, curr_elem.nodeStart, curr_elem.nodeEnd)
        # fanalysis = InternalForces(element, model)
        # println("fanalysis :", fanalysis)
        # # danalysis = ElementDisplacements(element, model)

        # println("axial forces :" ,axial_force.(model.elements))


        return model
    end

    """
    create_and_solve_model_julia(node_coords, element_conns, fixed_idx, loads, frame_length_m)

    Creates and solves a generic model using provided node coordinates, element connections, and applied loads. 
    Elements have varying thicknesses and strengths based on their types indicated through `element_types`.
    The function generates a `Model` in Julia, solves it, and returns the solved model. 
    From the model we get displacement, axial_forces, 
    Calculate the allowable stresses per element P_cap_kN based on yield strength (2/3 of axial capacity) and area of element type. 

    Inputs:
    - `node_coords::Vector{Pylist{Any}}`: 
        A vector of vectors representing the 2D or 3D coordinates of each node. 
        Each node coordinate is a vector of the form `[x, y]` or `[x, y, z]`.
        
    - `element_conns::Vector{Pylist{Any}}`: 
        A vector of vectors where each inner vector represents an element's connection between two nodes, given by their indices. 
        Each connection is of the form `[i, j]`, where `i` and `j` are the node indices (1-based).

    - `loads::Vector{Tuple{Int, Vector{Float64}}}`: 
        A vector of tuples where each tuple represents a load applied to a node. 
        The first element of the tuple is the node index (1-based), and the second element is a vector representing the load in each direction, e.g., `[Fx, Fy, Fz]`.

    - 'element_types::Vector{Tuple{Int, Vector{Float64}) 
        A vector of (element_type_int, outer diameter of tube, inward wall thickness in percentage) 

    - frame_length_m::Float64: 
        The actual length of the frame in meters. This is used to scale the node coordinates.

    Outputs
        - `displacement::Vector{Float64}`: 
            The computed displacements of the nodes in the model.
        - `axial_forces::Vector{Float64}`: 
            The computed axial forces in each element of the model.
        - `P_cap_kN:Vector{Float64}`: 
            The allowable axial force based on yield strength (2/3 of axial capacity).
        
    The model can be used to query results such as nodal displacements or reactions.
    """

    # function _create_and_solve_model_julia(node_coords, element_conns, fixed_idx, loads, frame_length_m)

    #     # Define Nodes using the custom Node structure
    #     # nodes = [Node(Vector{Float64}(coord), :fixed) for coord in node_coords]

    #     # nodes = [Node(Vector{Float64}(coord), i < 3 ? :pinned : :free) for (i, coord) in enumerate(node_coords)]
    #     # adjust node coordinates (node_coords) to have actual frame length (by default set to 2)
    #     node_coords_scaled = (node_coords ./ 2) .* frame_length_m
    #     nodes = [Node(Vector{Float64}(coord), i in fixed_idx ? :pinned : :free) for (i, coord) in enumerate(node_coords_scaled)]
        
    #     # Define material properties  (from 2D frame Asap test)
    #     E = 200e6 # Pa (Kn/m^2) young's modulus
    #     A = 0.00417 # m^2 Cross-sectional area of the element where is this from?? 
    #     I = 1.45686e-9 # m^4 Moment of inertia, used for bending capacity
    #     G = 80e6 # Shear modulus, also related to stiffness
    #     sec = Section(A, E, G, I, I, 1.) #area, young's modulus, shear mod, strong axis I, weak axis I, torsional constant, density=1

    #     # A = 0.001 # m^2 Cross-sectional area of the element
    #     # I = 1e-6 # m^4 Moment of inertia, used for bending capacity
    #     # Tube with radius 0.1 and 5% thickness inwards
    #     # A = 0.00314 # m^2 Cross-sectional area of the element
    #     # A = 0.00628 # m^2 Cross-sectional area of the element
        
    #     # Create Elements using the connections and custom Element structure
    #     elements = [Element(nodes[conn[1]], nodes[conn[2]], sec, release=:fixedfixed) for conn in element_conns] #default is fixedfixed

    #     # println("elements : ", elements )

    #     # Create Load objects based on the custom AbstractLoad structure
    #     loads = [NodeForce(nodes[load[1]], Vector{Float64}(load[2])) for load in loads]
    #     # println("loads: ", join(loads, "\n"))

    #     # Assemble the structural model
    #     model = Model(nodes, elements, loads)
        
    #     # Apply boundary conditions (DOFs) and build stiffness matrix
    #     planarize!(model)  # Assume this sets DOFs for planar structures
    #     solve!(model)  # Solves the model, computing displacements and reactions

    #     displacement = model.u

    #     # Output axial forces in elements
    #     # println("Axial forces: ", axial_force.(model.elements))
    #     axial_forces = axial_force.(model.elements)
    #     # yield strength for structural steel is around 250 MPa (or 250,000 Pa (kN/m²))
    #     fy = 350e3  # Yield strength in Pa(kN/m²) (assuming structural steel)
    #     # Calculate allowable axial force based on yield strength (2/3 of axial capacity)
    #     P_cap_kN = A * fy * 2 / 3  # kN 
    #     # println("P_cap_kN: ", P_cap_kN)
    #     # failed_elements = [i for (i, force) in enumerate(axial_forces) if abs(force) > P_cap_kN]

    #     # println("Elements that failed:", failed_elements)

    #     # return model
    #     return displacement, axial_forces, P_cap_kN
    # end

    # # """
    # # Given solved ASAP model, get element forces for each element in the model
    # # """
    # # function get_element_forces(model, increment=20)
    # #     return forces(model, increment)
    # # end

    """
    add element types to model 
    """

    function create_and_solve_model_julia(node_coords, 
                                            element_list_w_section, 
                                            fixed_idx, 
                                            loads, 
                                            frame_length_m, 
                                            )

        # Define Nodes using the custom Node structure
        # adjust node coordinates (node_coords) to have actual frame length (by default set to 2)
        node_coords_scaled = (node_coords ./ 2) .* frame_length_m
        nodes = [Node(Vector{Float64}(coord), i in fixed_idx ? :pinned : :free) for (i, coord) in enumerate(node_coords_scaled)]
        
        # Create list of elements (Element(node1, node2, section, release)) 
        # element_list is a list of tuples (node1, node2, section_outer_d, section_inner_thickness)
        # Constants
        E = 200e6  # Pa (Kn/m^2) young's modulus (steel)
        G = 80e6   # Pa Shear modulus, also related to stiffness (steel)
        # Function to compute a Section from outer diameter and thickness
        make_section(outer_d, thickness_percent) = begin
            A = (π * (outer_d^2 - (outer_d * (1 - thickness_percent))^2)) / 4 # m^2 Cross-sectional area of the element
            I = (π * (outer_d^4 - (outer_d * (1 - thickness_percent))^4)) / 64 # m^4 Moment of inertia, used for bending capacity
            Section(A, E, G, I, I, 1.0) #area, young's modulus, shear mod, strong axis I, weak axis I, torsional constant, density=1
        end
        
        # Map over the element list to create Element structs
        elements = map(elem -> begin
            v1_idx, v2_idx, outer_d, thickness_percent = elem
            sec = make_section(outer_d, thickness_percent)
            Element(nodes[v1_idx], nodes[v2_idx], sec, release = :fixedfixed)
        end, element_list_w_section)

        # Create Load objects based on the custom AbstractLoad structure
        loads = [NodeForce(nodes[load[1]], Vector{Float64}(load[2])) for load in loads]
        # println("loads: ", join(loads, "\n"))

        # Assemble the structural model
        model = Model(nodes, elements, loads)
        
        # Apply boundary conditions (DOFs) and build stiffness matrix
        planarize!(model)  # Assume this sets DOFs for planar structures
        solve!(model)  # Solves the model, computing displacements and reactions

        displacement = model.u

        # Output axial forces in elements
        # println("Axial forces: ", axial_force.(model.elements))
        axial_forces = axial_force.(model.elements)
        # yield strength for structural steel is around 250 MPa (or 250,000 Pa (kN/m²))
        fy = 350e3  # Yield strength in Pa(kN/m²) (assuming structural steel)

        # P_cap_KN is a list of allowable axial forces in order or element idx 
        # Calculate allowable axial force based on yield strength (2/3 of axial capacity)
        areas = [elements[i].section.A for i in 1:length(elements)] # list of areas for each element
        P_cap_kN = [(a * fy * 2 / 3) for a in areas] # kN

        # return model
        return displacement, axial_forces, P_cap_kN
    end

end