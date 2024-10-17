"""
Julia functions that use ASAP to run finite element analysis
"""

# truss_analysis.jl
module TrussAnalysis

export create_and_solve_truss_model_julia
export create_and_solve_model_julia

    using AsapToolkit
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
    create_and_solve_model_julia(node_coords, element_conns, loads)

    Creates and solves a generic model using provided node coordinates, element connections, and applied loads. 
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

    function create_and_solve_model_julia(node_coords, element_conns, fixed_idx, loads)

        # Define Nodes using the custom Node structure
        # nodes = [Node(Vector{Float64}(coord), :fixed) for coord in node_coords]
        """
        Common fixity types
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

        # nodes = [Node(Vector{Float64}(coord), i < 3 ? :pinned : :free) for (i, coord) in enumerate(node_coords)]
        nodes = [Node(Vector{Float64}(coord), i in fixed_idx ? :pinned : :free) for (i, coord) in enumerate(node_coords)]
        # println("nodes : ", nodes )

        
        # # Define material properties (from 2D truss Asap test)
        # E = 70.0  # Example Elastic modulus in kN/m^2
        # A = 4e3 / 1e6  # Example cross-sectional area in m^2
        # sec = Section(A, E)

        # Define material properties  (from 2D frame Asap test)
        E = 200e6
        A = 0.001
        I = 1e-6 #m^4
        G = 80e6
        sec = Section(A, E, G, I, I, 1.) #area, young's modulus, shear mod, strong axis I, weak axis I, torsional constant, density=1
        
        # Create Elements using the connections and custom Element structure
        elements = [Element(nodes[conn[1]], nodes[conn[2]], sec) for conn in element_conns] #default is fixedfixed
        # println("elements : ", elements )

        # Create Load objects based on the custom AbstractLoad structure
        loads = [NodeForce(nodes[load[1]], Vector{Float64}(load[2])) for load in loads]
        # println("loads : ", loads )

        # Assemble the structural model
        model = Model(nodes, elements, loads)
        
        # Apply boundary conditions (DOFs) and build stiffness matrix
        planarize!(model)  # Assume this sets DOFs for planar structures
        solve!(model)  # Solves the model, computing displacements and reactions

        # Output axial forces in elements
        # println("Axial forces: ", axial_force.(model.elements))

        return model
    end

end