import juliacall

# create system image with Asap and PythonCall to precompile packages so that there is no additional compilation at start of session
'''
using PackageCompiler
PackageCompiler.create_sysimage([:AsapToolkit], sysimage_path="custom_sysimage.so")

Precompilation in Julia means that Julia analyzes your package code and generates machine code for it ahead of time. 
This machine code is then stored in a cache, which reduces the time needed to load the package in the future. 
compiled_modules=True is default option in juliacall.newmodule() function.

When you call Julia from Python Julia will look for and load the precompiled modules (packages) from the precompilation cache. 
This means it still needs to spend some time checking for and loading the precompiled packages from disk into memory, which can lead to a slight startup delay.

compiled_modules=False: This option prevents Julia from loading the precompiled modules (i.e., it skips loading the cache) and forces Julia to interpret everything directly. 
This can reduce startup time slightly for certain cases where the overhead of loading the cache is higher than interpreting the package code. 
However, this usually results in slower runtime performance because package code will not be optimized.
'''

jl = juliacall.newmodule("SomeName") # compile_modules=False option?, sysimage-"*.so" option?

# jl.println("Hello from Julia!")

# x = jl.rand(range(10), 3, 5)
# x._jl_display()

# import numpy
# print(f' sum of matrix x is : {numpy.sum(x, axis=0)}')

# Check the current environment directory - it should be within conda environment 
curr_env = jl.seval('Base.active_project()')
print(f"The current active Julia environment is located at: {curr_env}")
# /Users/chong/opt/anaconda3/envs/trussframeASAP/julia_env/Project.toml
'''
activate the environment in Julia REPL
using Pkg
Pkg.activate("/Users/chong/opt/anaconda3/envs/trussframeASAP/julia_env")
'''

# Listing All Environments in a Julia Session: Show the current active environment:
all_active_env = jl.seval('Base.load_path()')
print(f"All active Julia environments : {all_active_env}")

# Develop the local package
# Pkg.develop(PackageSpec(path="/Users/chong/Dropbox/2024_Summer/TrussFrameASAP/AsapToolkit-main"))

# View all the packages in the current environment
jl.seval('using Pkg')
all_pkgs = jl.seval('Pkg.status()') # prints out all the packages in the current environment
'''
Status `~/opt/anaconda3/envs/trussframeASAP/julia_env/Project.toml`
  [763a8c06] Asap v0.2.1
  [45845293] AsapToolkit v0.1.0 `https://github.com/keithjlee/AsapToolkit.git#main`
  [6099a3de] PythonCall v0.9.23
'''

# instantiate the package (not needed if using system image)
jl.seval('Pkg.instantiate()') # download and install missing dependencies for packages to run 

# all_pkgs_dep = jl.seval('Pkg.status(; mode=PKGMODE_MANIFEST)')
# print(f"All packages added in the current environment: {all_pkgs_dep}")

# jl.eval('import AsapToolkit')
jl.seval('using AsapToolkit')
jl.seval('using Asap')

# test output of julia -> python 

# You can capture the output directly in Python
# section = jl.seval('toASAPframe(rand(allW()), 200e3, 80e3)')
# Asap.Section(44200.0, 200000.0, 80000.0, 1.25e9, 4.79e8, 2.48e7, 1.0)

# Now `section` is a Python object containing the result from Julia
# print("The section returned from Julia is:", section)

# Internal Analysis Example
# internal_analysis = """
# using Asap
# section = toASAPframe(rand(allW()), 200e3, 80e3)

# n1 = Node([0., 0., 0.], :fixed)
# n2 = Node([6000., 0., 0.], :fixed)
# nodes = [n1, n2]

# element = Element(n1, n2, section)
# elements = [element]

# load1 = LineLoad(element, [0., 0., -30])
# pointloads = [PointLoad(element, rand(), 25e3 .* [0., 0., -rand()]) for _ = 1:5]
# loads = [load1; pointloads]

# model = Model(nodes, elements, loads)
# solve!(model)

# fanalysis = InternalForces(element, model)
# danalysis = ElementDisplacements(element, model)
# """

# # Execute the Julia code block
# jl.seval(internal_analysis)
# '''
# ERROR!
# juliacall.JuliaError: type Element has no field release 
# '''

# # Now retrieve the values of fanalysis and danalysis in Python
# fanalysis = jl.eval('fanalysis')  # Access the Julia variable fanalysis
# danalysis = jl.eval('danalysis')  # Access the Julia variable danalysis

# # Now fanalysis and danalysis can be used in Python
# print(f"Internal Forces Analysis: {fanalysis}")
# print(f"Element Displacements Analysis: {danalysis}")

# Generation Example
# generation = """
# using Asap

# #parameters
# n = 11
# dx = 1500
# dy = 1400
# section = toASAPtruss(rand(allHSSRound()), 200e3)

# #generate
# warrentruss = Warren2D(n,
#     dx,
#     dy,
#     section)

# #extract
# truss = warrentruss.model

# #print truss
# println(truss)

# # extract needed information for Python
# """

# '''
# Q : does model = solved? or is there separate solving step?
# '''

# jl.seval(generation) # execute generation
# truss = jl.eval('truss') # retrieve the truss model
# print(f"Truss Model: {truss}")

# [WORKED] run text example
runtext = """
tol = 0.1
# 2D truss test: Example 3.9 from Kassimali "Matrix Analysis of Structures 2e"
# in kips, ft

n1 = TrussNode([0., 0., 0.], :fixed)
n2 = TrussNode([10., 0., 0.], :fixed)
n3 = TrussNode([0., 8., 0.], :yfree)
n4 = TrussNode([6., 8., 0.], :free)

nodes = [n1, n2, n3, n4]

E = 70. #kN/m^2
A = 4e3 / 1e6 #m^2

sec = TrussSection(A, E)

e1 = TrussElement(nodes[[1,3]]..., sec)
e2 = TrussElement(nodes[[3,4]]..., sec)
e3 = TrussElement(nodes[[1,4]]..., sec)
e4 = TrussElement(nodes[[2,3]]..., sec)
e5 = TrussElement(nodes[[2,4]]..., sec)

elements = [e1,e2,e3,e4,e5]

l1 = NodeForce(n3, [0., -400., 0.])
l2 = NodeForce(n4, [800., -400., 0.])

loads = [l1, l2]

model = TrussModel(nodes, elements, loads)
planarize!(model)
solve!(model)

reactions = model.reactions[model.fixedDOFs]
reactions2d = reactions[reactions .!= 0]
"""

jl.seval(runtext) # execute generation
# Extract the Julia variables into Python
reactions = jl.seval("reactions")  # Extract the 'reactions' variable
reactions2d = jl.seval("reactions2d")  # Extract the 'reactions2d' variable

# Print or use the extracted values in Python
print("Reactions:", reactions)
print("Reactions 2D:", reactions2d)



# TODO how to visualize?