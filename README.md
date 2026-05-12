# AFM_Youngs_Modulus
Get the radial Young's modulus and stress of a membrane from AFM poking data.

Code to simulate the behavior of a circular suspended membrane when poked with an AFM tip. Includes fitting Young's modulus and tension of the membrane to experimental data.

Run `afm_youngs_modulus_master.py`.

---

## Workflow overview

The analysis consists of three sequential steps, each accessible from the main window:

### Step 1 — Data Processor
Load raw AFM force-map data, fit the compliance of each pixel, clean up the resulting 2D compliance map, and project it onto a 1D radial profile. Save the result as an `.npz` file.

### Step 2 — Compliance Fit (FEniCS)
Load the `.npz` file produced in Step 1. Use a finite-element simulation of the membrane to compute the expected radial compliance profile and fit it to your data by varying the bending rigidity and/or tension. This gives you Young's modulus and pre-tension.

### Step 3 — Cubic Fit (optional)
An alternative / cross-check approach that fits the raw force-deflection curve at a single point on the membrane with an analytical model (linear + cubic term), extracting Young's modulus directly from the force curve shape.

---

## Step 1: Data Processor

### Loading data

**Directory browser** — Select the folder containing all the raw AFM text files. The code identifies files by two substrings in their filenames:

- **X-name** — substring that identifies the z-sensor position files (x-axis of the force curve, e.g. `Zsensor`).
- **Y-name** — substring that identifies the tip-deflection files (y-axis of the force curve, e.g. `Vertical`).

Files are sorted automatically by their pixel grid position (row, column) extracted from the filename numbers.

**Tip spring constant (k_tip)** — The calibrated spring constant of the AFM cantilever in N/m. This converts the measured tip deflection (m) into force and thereby into an absolute compliance value (m/N). The compliance of each pixel is computed as `(1/slope_average - 1) / k_tip`. If you change this value, the Fit Compliance button turns red to remind you to refit.

**Fit Compliance button** — Fits a line to both the approach and retract parts of every force curve and extracts the slope. The average of approach and retract slopes is used to fill the 2D compliance map.

---

### Smoothing the compliance map

Raw compliance maps often contain outlier pixels (failed fits, tip-sample adhesion artefacts, pixels with zero compliance). The smoothing step replaces these with interpolated values from their neighbours.

**Threshold** — Any pixel whose absolute compliance exceeds this value (in m/N, but think of it in relative terms compared to your data range) is treated as an outlier and replaced. Pixels that are exactly zero are also always replaced. Start with a value slightly above the maximum realistic compliance of your membrane; reduce it if artefacts remain, increase it if too many good pixels are being removed.

**Smoothing method:**

- **Nearest Neighbor** — Replaces each outlier pixel with the simple average of its four direct neighbours (up, down, left, right). A second pass then checks all pixels for consistency with their neighbours (threshold 0.002 m/N). This is fast and works well when outliers are sparse. It does not touch non-outlier pixels at all.

- **Gaussian** — Applies a NaN-aware Gaussian blur over the entire map. Outlier pixels are first masked as NaN so they do not contaminate their neighbours during filtering; the Gaussian kernel is renormalised around the missing pixels. This produces a smoother result and is better when many adjacent pixels are bad, but it will soften sharp features in the compliance map.

**Sigma** (Gaussian only) — Standard deviation of the Gaussian kernel in pixels. Larger sigma = stronger smoothing and more blurring of real features. A value of 1–2 pixels is a reasonable starting point for typical AFM maps of 32×32 or 64×64 pixels.

---

### Setting the circle geometry

The compliance map is overlaid with two interactive red markers:

- **Center marker** — Drag to the center of the suspended membrane. Alternatively, use **Find Center** which attempts to locate the maximum-compliance region automatically from the current map.
- **Radius marker** — Drag to the edge of the membrane. The radius in pixels and in µm (computed from the scan window size) updates live.

**Scan window size (µm)** — The physical size of the entire AFM scan in µm (e.g. 10 for a 10 µm × 10 µm scan). This is needed to convert pixel distances to physical distances.

All subsequent radial projections use the circle center and radius you set here. The compliance is expressed as a function of `r/r₀` where `r` is the distance from the center and `r₀` is the membrane radius, so `r/r₀ = 1` is the clamped edge.

---

### Creating radial compliance data

**Project Radial Compliance** — Flattens the 2D map into a scatter plot of compliance vs. `r/r₀`. Every pixel is assigned a distance from the center; no averaging is performed here. Use this to visually inspect the data quality before averaging.

**Averaging method:**

- **Polar** — Converts to polar coordinates centred on the membrane. For each radial distance the code interpolates the compliance map onto a ring of 80 evenly spaced angles using a 2D spline and averages over the ring. This correctly accounts for the geometry of the circular membrane and produces smooth, well-sampled radial profiles. It is the recommended method.

- **Bin** — Divides the radial range 0–1 into equal-width bins and averages all pixels that fall into each bin. Simpler and faster, but the number of pixels per bin varies (fewer near the center), so the innermost bins can be noisy.

**Number of radial points** — How many discrete radial positions to compute (Polar) or how many bins to use (Bin). More points give a smoother curve but are slower for Polar. 20–40 is typically sufficient.

**Zero compliance bounds (min, max)** — The radial range (in units of `r/r₀`) used to estimate the baseline compliance of the clamped substrate. Points in this range are averaged and subtracted from all compliance values so that the clamped edge sits at zero. Values above 1 sample the substrate outside the membrane. A range like `[1.05, 1.2]` works well for most geometries; make sure there is actual scan data in that range.

**Create Ave Data button** — Computes the averaged radial profile and overlays it (orange) on the raw scatter (white).

---

### Saving

**Save All** — Saves an `.npz` file containing the compliance map, the raw and averaged radial compliance data, circle center, radius in pixels and µm, scan window size, and tip spring constant. This file is the input for Step 2.

---

## Step 2: Compliance Fit

This module uses FEniCS (via DOLFINx) to solve the biharmonic plate equation for a clamped circular membrane and computes the expected compliance as a function of radial position. It then fits the simulation to your experimental data.

### Mesh parameters

**Mesh radius** — Radius of the simulated disc in µm. For best results keep this equal to the experimental radius, but the simulation is always run in normalised coordinates internally so the exact value matters less than the mesh resolution.

**Characteristic mesh length** — Controls the density of the finite-element mesh. Smaller values give a finer mesh (more accurate but slower). A value around 0.05–0.1 for a radius-1 disc is a good starting point. You need to click **Update Mesh** after changing either mesh parameter.

### Physical parameters

These must be updated with **Update Params** before running the simulation.

**Experimental radius (µm)** — The measured radius of your membrane. Loaded automatically from the `.npz` file. Used to convert between normalised and physical coordinates.

**Bending rigidity (N·pm)** — The flexural rigidity `D = E t³ / (12(1−ν²))` of the membrane. Linked to Young's modulus: changing either bending rigidity or Young's modulus updates the other automatically given the thickness and Poisson ratio.

**Young's modulus (GPa)** — Derived from bending rigidity, thickness, and Poisson ratio via `E = 12D(1−ν²)/t³`. This is the quantity you ultimately want to measure.

**Thickness (nm)** — Mechanical thickness of the membrane. Must be known from another measurement (e.g. AFM step-height). Together with Poisson ratio it links bending rigidity to Young's modulus.

**Poisson ratio** — Dimensionless ratio of transverse to axial strain. Typically 0.16 for graphene, ~0.3 for many metals, ~0.5 for rubbers. Affects the relationship between bending rigidity and Young's modulus.

**Tension (N/m)** — Pre-tension (in-plane stress resultant) of the membrane. Suspended membranes are typically under tensile pre-stress from fabrication. This adds a membrane-stiffness term to the compliance on top of the bending stiffness. Both bending rigidity and tension can be varied simultaneously in the fit.

**Tip size (nm)** — Effective radius of the AFM tip contact area. The load is distributed over this area in the simulation rather than applied as a point force.

**Force (N)** — Magnitude of the applied force in the simulation. Should match the typical force used in your experiment. Compliance is force-independent for a linear system, but this value is needed to get absolute displacement scales right.

**Edge spring constant** — Stiffness of a soft spring at the clamped boundary, used to model any residual compliance at the edge (e.g. the membrane is not perfectly clamped). Expressed in internal units (displayed as multiples of 10¹³). Set to zero for a perfectly clamped boundary.

### Running the simulation and fit

**Calculate radial compliance** — Runs the FEniCS simulation for each radial position and computes the compliance profile. The number of radial points is set by **Number of points**.

**Fit data source** — Choose whether to fit the **Ave Data** (averaged radial profile from Step 1) or the **Raw Data** (all individual pixel values). Ave data is smoother and faster to fit; raw data uses all available information.

**Fit Data** — Runs a least-squares fit (via lmfit) minimising the difference between simulation and data. The parameters that are marked `vary=True` in the lmfit Parameters object (bending rigidity and tension by default) are adjusted. The fit updates the parameter fields live as it runs.

**Calculate compliance map** — Runs a single FEniCS solve at the load point you specify (x-coord, y-coord in mesh coordinates) and shows the 2D displacement field. Useful for a visual sanity check.

---

## Step 3: Cubic Fit

An analytical alternative that fits the force-deflection curve at a single pixel with:

```
F(δ) = (4π E_lin t³)/(3(1−ν²)R²) + π T) · δ  +  c · E_cub · t/R² · δ³
```

The linear term captures bending stiffness and pre-tension; the cubic term captures geometric nonlinearity (membrane stretching) at large deflections.

### Loading data

Load the **compliance map** (`.npz` from Step 1) first — this sets the circle center, radius, and k_tip automatically. Then browse for the **raw data folder** (same folder you used in Step 1) using the same X-name / Y-name identifiers.

Click on a pixel in the compliance map to select a measurement point, then **Plot Selected Data** to display the approach and retract force curves.

### Selecting the fit range

Two movable vertical lines define the range of the force curve used for fitting. Drag them to bracket the contact region you want to fit — typically the linear-to-nonlinear transition after the snap-in. The selected data is shown in the cubic fit plot.

**Fit data** — Choose whether to fit the **approach** or **retract** curve.

### Physical parameters

**Thickness (nm)** — Membrane thickness (same as in Step 2).

**Poisson ratio** — Same as in Step 2.

**Tension (N/m)** — Pre-tension. If unknown, set to zero or use the value from Step 2 as a starting point.

### Fit parameters

Each parameter has an initial guess field and a checkbox to choose whether it is varied in the fit or held fixed.

**E_lin (GPa)** — Young's modulus entering the linear (bending + tension) term. For thin membranes at small deflections this dominates; it is related to the bending rigidity extracted in Step 2.

**E_cub (GPa)** — Young's modulus entering the cubic (stretching) term. This becomes important at large deflections where the membrane is being stretched rather than just bent. For an isotropic membrane E_lin and E_cub should converge to the same value in the appropriate limits.

**x_shift** — Small horizontal offset of the deflection origin in the force curve. Accounts for the fact that the contact point (δ=0) cannot be picked perfectly. Should remain small; a large x_shift indicates a poor range selection.

**y_shift** — Small vertical offset of the force origin. Same idea as x_shift but for the force axis.

**Fit Cubic** — Runs the least-squares fit. Results are shown in the result fields below.

---

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
# pip install pyqt5==5.15.11 PyOpenGL==3.1.10 PyOpenGL_accelerate==3.1.10 pyqtgraph==0.14.0 gmsh==4.15.2 lmfit==1.3.4 pyvistaqt==0.11.4 pandas==3.0.2 opencv-python==4.13.0
pip install pyqt5 PyOpenGL PyOpenGL_accelerate pyqtgraph gmsh lmfit pyvistaqt pandas opencv-python
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
