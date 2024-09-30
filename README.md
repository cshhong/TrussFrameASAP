# TrussFrameASAP: User-Playable Version

This repository provides a user-playable version of TrussFrameASAP, designed to create and test a Reinforcement Learning (RL) environment. In this environment, modular structural frames are sequentially built and analyzed using a Finite Element Analysis (FEA) tool ASAP, which is implemented in Julia. The overarching project aims to train an RL agent that can interactively design and optimize frame structures in terms of constructability and structural performance. 

## Files Overview

- **main.py**: 
  - The primary interface for the user. It creates a drawing board where users can sequentially add truss frames by clicking on a grid. The interface also visualizes the progressive displacement of the structure as it undergoes FEA.
  
- **maximaledge.py**: 
  - Processes geometric representations of TrussFrames into a graph structure, converting them into nodes and edges that can be utilized for FEA.
  
- **env_cantilever.py**: 
  - Generates random cantilever boundary constraints, contributing to the RL environment setup for training and testing.

- **graph.py** & **vertex.py**: 
  - Custom data structures that store additional information required for performing FEA on the truss structures.

- **pythonAsap.py**: 
  - A Python wrapper used to communicate input and output between the Python components and the Julia module ASAP, facilitating integration between the two.

- **truss_analysis.jl**: 
  - A Julia script that runs the ASAP Finite Element Analysis, processing the structural data from Python to compute displacements and other results.



