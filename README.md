# AFM_Youngs_Modulus
Get the radial Young's modulus and stress of a membrane from AFM poking data.

Code to simulate the behavior of a circular suspended membrane when poked with an AFM tip.\
Includes fitting Young's modulus and tension of the membrane to experimental data

Run "afm_youngs_modulus_master.py".

## Requirements

Tested with **Python 3.14.4** inside a conda environment. May need to update conda and/or Anaconda before starting.

### 1. Create the conda environment and install FEniCSx

The most integral part of this code is based on the [FEniCS project](https://fenicsproject.org/).

```bash
conda create -n afm python=3.14
conda activate afm
conda install -c conda-forge fenics-dolfinx=0.10.0 mpich mpi4py pyvista
```

### 2. Install remaining packages via pip

```bash
pip install pyqt5==5.15.11 PyOpenGL==3.1.10 PyOpenGL_accelerate==3.1.10 pyqtgraph==0.14.0 gmsh==4.15.2 lmfit==1.3.4 pyvistaqt==0.11.4 pandas==3.0.2 opencv-python==4.13.0
```

### Full package version reference

| Package | Version | Install via |
|---|---|---|
| python | 3.14.4 | conda |
| fenics-dolfinx | 0.10.0 | conda |
| fenics-basix | 0.10.0 | conda |
| fenics-ufl | 2025.2.1 | conda |
| mpi4py | 4.1.1 | conda |
| mpich | (conda-forge default) | conda |
| pyvista | 0.47.3 | conda |
| numpy | 2.4.4 | conda (dependency) |
| scipy | 1.17.1 | conda (dependency) |
| matplotlib | 3.10.9 | conda (dependency) |
| PyQt5 | 5.15.11 | pip |
| PyOpenGL | 3.1.10 | pip |
| pyqtgraph | 0.14.0 | pip |
| gmsh | 4.15.2 | pip |
| lmfit | 1.3.4 | pip |
| pyvistaqt | 0.11.4 | pip |
| pandas | 3.0.2 | pip |
| opencv-python | 4.13.0 | pip |

## API changes from older dolfinx versions

The code was updated to work with **dolfinx 0.10.0**. If you hit import or runtime errors with a different version, the two breaking changes were:

1. **`gmshio` renamed to `gmsh` inside `dolfinx.io`**
   - Old: `from dolfinx.io import gmshio`
   - New: `from dolfinx.io import gmsh as dolfinx_gmsh`

2. **`model_to_mesh` now returns a `MeshData` named tuple** (6 fields) instead of a 3-tuple
   - Old: `msh, _, _ = gmshio.model_to_mesh(...)`
   - New: `msh = dolfinx_gmsh.model_to_mesh(...).mesh`

3. **`petsc.LinearProblem` requires `petsc_options_prefix`** as a keyword argument
   - Old: `petsc.LinearProblem(a, L, u=sol)`
   - New: `petsc.LinearProblem(a, L, u=sol, petsc_options_prefix="membrane_")`
