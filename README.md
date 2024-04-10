SCRYiNG (Simulated CRYstal Nucleation and Growth) is a Python package for the simulation of 2-dimensional crystal growth. 

### Overview
SCRYiNG uses a modified kinetic Monte Carlo simulation with alternating nucleation and growth steps. An intuitive and highly customizable object-oriented framework (`scrying.py`) is provided for scripting. The simulation parameters (e.g. image size, simulation time, nucleation events per time step, etc.) can be easily reconfigured. SCRYiNG also supports specification of irregular crystal shapes (any convex polygon) and the importing of nucleation data (such as those extracted from experiments) for use in simulations. 

A variety of example uses are shown in `examples.ipynb`. Documentation of all publicly accessible features can be found in the comments. 

### User Interface
A user interface (UI) is also provided to facilitate use by those without programming experience. This UI (`scrying-ui.py`) offers most of the features available in the scripting version within a simple application. 

The user interface version can be run from a command line via `python scrying-ui.py`. Examples of the properly formatted data for importing custom crystal shapes and nucleation data are provided in the `/ui/` folder. 

### Requirements 
*All package versions listed below were the versions used during development.* 

**Core**
* NumPy (1.20.1)
* SciKit-Image (0.19.2)

Data visualization and storage libraries (e.g. matplotlib, tifffile, array2gif) are also recommended for exporting results. 

**User Interface**
* ttkwidgets (0.13.0)
* Matplotlib (3.5.0)
* Tifffile (2022.3.16)
* Array2GIF (1.0.4)



