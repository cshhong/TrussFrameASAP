## Repository Overview

### **gymenv/**
Custom gym environments
- `__init__.py`: Registers custom environments to be detected by the Gymnasium module.
- `cantileverenv_convert_gymspaces.py`: Encodes and decodes Gym observation and action spaces for the Cantilever-v0 environment.
- `cantileverenv_torchrl.py`: Implements the Cantilever environment as an EnvBase class for TorchRL.
- `cantileverenv_v0.py`: Implements the Cantilever environment as a Gymnasium Env class.

---

### **humanplayable/**
Human playable version of TrussFrame Environment to help debugging. 
- `main.py`: The primary user interface creates a matplotlib drawing board where user can sequentially add truss frames by clicking on a grid. Visualizes the progressive displacement of the structure during FEA.
- `pythonAsap_human.py`: A Python wrapper for communication between Python and Julia.Facilitates the exchange of (Python) FEAGraph → (Julia) ASAP FEA → (Python) displacement in `cantileverenv_v0.py`.

---

### **PerformanceMap/**
Save rollout and visualize in performance map.
- `h5_utils.py`: Functions for storing and querying data from HDF5 files.
- `perfmap.py`: Uses random rollout data for one boundary condition to create a 3D performance map.

---

### **TrussFrameMechanics/**
Mechanics of TrussFrame Environment. 
- `__init__.py`: Ensures that `TrussFrameMechanics` is recognized as a Python package.
- `feagraph.py`: Defines a custom graph data structure (FEAGraph) for FEA.
- `generate_bc.py`: Functions for generating boundary conditions (supports, external loads) within the environment.
- `maximaledge.py` (not used): Processes geometric representations of TrussFrames into graph structures for use in FEA.
- `pythonAsap.py`: A Python wrapper for communication between Python and Julia.Facilitates the exchange of (Python) FEAGraph → (Julia) ASAP FEA → (Python) displacement in `cantileverenv_v0.py`.`cantileverenv_v0.py`.
- `truss_analysis.jl`: A Julia script for running Finite Element Analysis with ASAP. Processes structural data from Python to compute displacements and other results.
- `trussframe.py`: Defines the `TrussFrame` object and `FrameShapeType`.
- `vertex.py`: Custom `Vertex` data structure storing IDs, coordinates, and edge information.

---

### **misc/**
- `problemsize.py`: Calculates the Monte Carlo tree size of the RL environment.
- `test_rollout-livedemo.py`: Creates random rollouts and generates a performance map.
- `julia_env_configure.py`: Configures the Julia module with AsapToolkit installed.
