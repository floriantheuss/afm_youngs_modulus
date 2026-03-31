# AFM_Youngs_Modulus
Get the radial Young's modulus and stress of a membrane from AFM poking data.

Code to simulate the behavior of a circular suspended membrane when poked with an AFM tip.\
Includes fitting Young's modulus and tension of the membrane to experimental data

## Requirements
- only tested with python 3.12 (known not to work on python 3.9)
- may need to update conda and/or anaconda
This is python based code, most easily run if install with an Anaconda distribution.\
The most integral part of this code is based on the [fenics project](https://fenicsproject.org/). It can be installed using conda (see website for details):
- conda create -n fenicsx-env
- conda activate fenicsx-env
- conda install -c conda-forge fenics-dolfinx mpich pyvista

Other packages required to run this code (most can be installed using "**pip install ...**" - obviously not the most elegant to switch between conda and pip, but it work ...):
- pyqt5
- PyOpenGL
- pyqtgraph
- gmsh
- lmfit
- pyvistaqt
- pandas
- opencv-python