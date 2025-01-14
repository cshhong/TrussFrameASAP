'''
Create julia module and install AsapToolkit
'''
import juliacall
jl = juliacall.newmodule("TrussFrameRL") 
print(jl.seval('Base.active_project()'))

jl.seval('using Pkg')
# jl.seval('Pkg.add("Asap")')

jl.seval('import Pkg; Pkg.add("AsapToolkit")')