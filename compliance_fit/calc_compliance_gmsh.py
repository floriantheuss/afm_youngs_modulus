import gmsh
import dolfinx
from dolfinx.io import gmshio, XDMFFile
from dolfinx import fem, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import pyvista
import numpy as np
import ufl
from ufl import exp, inner, nabla_grad, dx, ds
from basix.ufl import element, mixed_element
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import platform

class CalcCompliance:
    def __init__ (self, exp_radius=1):
        """
        the entire simulation can be done in two ways:
        1) we just give every length in microns and then if we do differently sized holes we have to mesh the hole with the correct radius in microns;
        2) we give every length in units of the radius of the whole; in that case we can always keep the the meshed circle at radius 1 and only have to rescale parameters;
        - if exp_radius=1 then we are in the first case - but don't forget to remesh the circle then!;
        - for the second case, put exp_radius=measured radius of the circle in um
        """
        self.operating_system = platform.system()
        if self.operating_system in ['windows', 'Windows']:
            self.separator = '\\'
        elif self.operating_system in ['mac', 'Mac', 'Darwin', 'darwin']:
            self.separator = '/'
        else:
            print('operating system not any of the possible options')
            print('current operating system is: ', self.operating_system)

        self.mesh = None
        # fenicsx function space variables
        self.W, self.U, self.V = None, None, None
        self.u, self.f = None, None
        self.v, self.w = None, None
        self.x     = None
        self.Vproj = None
        
        self.current_u_fct = None # latest solution to the finite element problem
        self.exp_radius    = exp_radius # experimental radius of the drumhead
        self.progress      = 0 # when doing the radial compliance calculation, this number keeps track of how many points are already calculated
                               # the only purpose of this attribute is so that we can update a progress bar in the gui


      
    def generate_disc_msh(self, radius=1, center=[0,0,0], msh_length=0.1):
        """use gmsh to generate mesh for a circle:
        - radius: radius of circle
        - center: position of the center of the circle - keep center[2]=0 to keep consistent with simulations below
        - msh_length: characteristic length of each triangle corner in the mesh: good value is 0.03 for radius=1 to get good resolution
        """
        gmsh.initialize()
        gmsh.model.add("Circle")
        # Define circle geometry (center at (0, 0), radius 1)
        circle = gmsh.model.occ.addDisk(xc=center[0], yc=center[1], zc=center[2], rx=radius, ry=radius)

        # Synchronize the model
        gmsh.model.occ.synchronize()

        # Define a physical group for the surface
        gmsh.model.addPhysicalGroup(dim=2, tags=[circle], tag=1)
        gmsh.model.setPhysicalName(dim=2, tag=1, name="Circle")

        # Specify the msh size
        # msh_length = relative_msh_length * radius # Characteristic length (smaller values give finer mshes)
        gmsh.model.mesh.setSize(dimTags=gmsh.model.getEntities(0), size=msh_length)

        # Generate the msh
        gmsh.model.mesh.generate(dim=2)
        # gmsh.finalize()
        # Convert Gmsh model to Dolfinx msh
        msh, _, _ = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

        # Finalize Gmsh
        gmsh.finalize()

        self.mesh = msh
        return msh
    
    def prepare_fenicsx_function_space (self, msh):
        """
        fenicsx needs function spaces and test and trial functions defined on the mesh;
        this fct needs to be run before self.set_up_and_solve_finite_element_problem;
        we are separating this out because it only needs to be run once for each mesh and then many simulations can be run on the same mesh
        msh: mesh these attributes are defined on
        """
        # create function spaces with order 1 Lagrangians on msh
        self.W = element("Lagrange",msh.basix_cell(), 1)
        self.U = element("Lagrange",msh.basix_cell(), 1)
        V = mixed_element([self.W,self.U])
        self.V     = fem.functionspace(msh,V)
        self.Vproj = fem.functionspace(msh,self.U)

        self.u, self.f = ufl.TrialFunctions(self.V)
        self.v, self.w = ufl.TestFunctions(self.V)
        self.x   = ufl.SpatialCoordinate(msh)
        return self.W, self.U, self.Vproj, self.V, self.u, self.f, self.v, self.w, self.x

    def set_up_and_solve_finite_element_problem (self, msh, point, D, tip, force, T, k, fenicsx_params):
        """
        set up variational problem to find deflection map of drumhead with Gaussian load at certain point;
        see Varun's thesis for details
        msh: mesh of the drumhead - make sure that circle is meshed in x-y space
        point: point in 2D (i.e. [x,y]) of where the force is applied
        h: thickness of the membrane in um
        E: Young's modulus of the membrane in N/um^2
        nu: Poisson ratio of the membrane
        tip: radius of the AFM tip pushing on the membrane in um
        force: force of the tip in N
        T: tension of the membrane in N/um
        k: spring shear constant on the boundary
        fenicsx_params: touple containing: square functionspace V, trial functions u and f, test functions v and w, fenicsx spatial coordinates x
        notation in fenicsx_params is consistent with that in self.prepare_fenicsx_function_space"""

        # all variables need to be defined as functions on the mesh
        # all length scales are rescaled to self.exp_radius
        force = fem.Constant(msh, force) # loading force
        # h     = fem.Constant(msh, h/self.exp_radius) # membrane thickness
        tip   = fem.Constant(msh, tip)#/self.exp_radius) # tip radius
        T     = fem.Constant(msh, T*self.exp_radius) # tension
        k     = fem.Constant(msh, k) # edge shear spring const
        # E     = fem.Constant(msh, E*self.exp_radius**2) #Youngs modulus
        # nu    = fem.Constant(msh, nu) #poisson ratio
        # D     = E * h**3 / (12 * (1 - nu**2)) #bending rigidity
        D     = fem.Constant(msh, D/self.exp_radius) # bending rigidity
        P   = force/(3.14*tip) # loading pressure

        px    = fem.Constant(msh, point[0])
        py    = fem.Constant(msh, point[1])

        V, u, f, v, w, x = fenicsx_params
        
        # the pressure of the AFM tip is defined with a Gaussian profile
        p = P*exp(-((x[0]-px)**2 + (x[1]-py)**2) / tip)

        # Weak form of the equation (again see thesis for details)
        a = (-(D) * inner(nabla_grad(u), nabla_grad(v))-T*u*v-inner(nabla_grad(f),nabla_grad(w))-f*v) * dx + k*u*w*ds
        L = p* w * dx

        sol = fem.Function(V) # need to define a new function on the function space
        problem = petsc.LinearProblem(a, L, u=sol) 
        problem.solve()

        # solution is defined on the function space V (which is square of the mesh because we are solving two equations)
        # to plot it on the mesh, we need to project it down to be only on the mesh (not the square)
        u_sol = sol.sub(0)
        f_sol = sol.sub(1)
        # V_u = fem.functionspace(msh, W)
        # V_f = fem.functionspace(msh, U)

        # # Project the sub-functions onto the new function spaces
        # u_proj = fem.Function(V_u)
        # f_proj = fem.Function(V_f)

        # u_proj.interpolate(u_sol)
        # f_proj.interpolate(f_sol)
        
        self.current_u_fct = u_sol
        return u_sol
    
    # def calculate_deflection_map (self, msh, point, h, E, nu, tip, force, T, k):
    #     _, _, _, V, u, f, v, w, x = self.prepare_fenicsx_function_space(msh)
    #     fenicsx_params = (V, u, f, v, w, x)
    #     u_sol = self.set_up_and_solve_finite_element_problem (msh, point, h, E, nu, tip, force, T, k, fenicsx_params)
    #     return u_sol


    # def plot_color_map_pyvista (self, fenicsx_fct, msh):
    #     """
    #     evaluate a function on a fenicsx function space over the entire mesh
    #     """
    #     W = element("Lagrange",msh.basix_cell(), 1)
    #     V = fem.functionspace(msh, W)

    #     f_proj = fem.Function(V)
    #     f_proj.interpolate(fenicsx_fct)
    #     val_map = f_proj.x.array.real
    #     dat = val_map/max(abs(val_map))

    #     cells, types, x = dolfinx.plot.vtk_mesh(V)
    #     grid = pyvista.UnstructuredGrid(cells, types, x)
    #     grid.point_data["u"] = dat
    #     grid.set_active_scalars("u")
    #     plotter = pyvista.Plotter()
    #     plotter.add_mesh(grid, show_edges=True)
    #     warped = grid.warp_by_scalar()
    #     plotter.add_mesh(warped)
    #     plotter.show()
    #     return 1

    # def plot_color_map_matplotlib (self, fenicsx_fct, msh):
    #     """
    #     evaluate a function on a fenicsx function space over the entire mesh
    #     """
    #     W = element("Lagrange",msh.basix_cell(), 1)
    #     V = fem.functionspace(msh, W)
        
    #     f_proj = fem.Function(V)
    #     f_proj.interpolate(fenicsx_fct)
    #     val_map = f_proj.x.array.real*self.exp_radius
    #     dat = val_map/max(abs(val_map))

    #     cells, types, x = dolfinx.plot.vtk_mesh(V)
    #     triangles = cells.reshape(-1, 4)[:, 1:]

    #     # Create a Triangulation object
    #     triangulation = Triangulation(x[:, 0], x[:, 1], triangles)

    #     # Plot the mesh and the scalar data
    #     plt.figure()
    #     # plt.tricontourf(triangulation, dat, cmap='viridis')
    #     plt.tripcolor(triangulation, dat, shading='flat', cmap='viridis')
    #     plt.colorbar(label='Scalar Data')
    #     # plt.triplot(triangulation, 'ko-', lw=0.5)
    #     plt.triplot(triangulation, '-', lw=0.5)
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.title('2D Mesh with Scalar Data')
    #     plt.show()
    #     return 1

    # def get_colormap_plot_data_pyqtgraph (self, Vproj, fenicsx_fct=None):
    #     """
    #     """
    #     # Extract the mesh points, cells, and types
    #     cells, types, points = dolfinx.plot.vtk_mesh(Vproj)
    #     triangles = cells.reshape(-1, 4)[:, 1:]
    #     triangles = triangles[types == 5]

    #     if fenicsx_fct is not None:
    #         f_proj = fem.Function(Vproj)
    #         f_proj.interpolate(fenicsx_fct)
    #         val_map = f_proj.x.array.real*self.exp_radius
    #     else:
    #         val_map = None

    #     return points, triangles, val_map
    
    def get_colormap_plot_data_pyvista (self, Vproj, fenicsx_fct=None):
        """
        get everything you need to create a plot of the mesh and the data in a pyvista plot;
        - Vproj: function space on the mesh (but not a square one!!!)
        - fenicsx_fct: a fenicsx function defined on Vproj"""
        if fenicsx_fct is not None:
            f_proj = fem.Function(Vproj)
            f_proj.interpolate(fenicsx_fct)
            # here is where we rescale the values of the function with self.exp_radius, to recover the correct values in um
            # we need to do this because we had rescaled all length scales to self.exp_radius in self.set_up_and_solve_finite_element_problem
            val_map = f_proj.x.array.real*self.exp_radius
        else:
            val_map = None

        cells, types, x = dolfinx.plot.vtk_mesh(Vproj)
        
        return cells, types, x, val_map
    
    def extract_values_at_points (self, fenicsx_fct, msh, points):
        """
        extract values of a funtion defined on a fenicsx function space at certain points;
        the points do not have to be points on the mesh;
        fenicx_fct: function defined within fenicsx
        msh: mesh over which the function is defined
        points: array where each element is [x,y,z] coordinates
        """
        tol = 0.001  # Avoid hitting the outside of the domain
        points = np.array(points, dtype='double')
        bb_tree = geometry.bb_tree(msh, msh.topology.dim)
        cells = []
        points_on_proc = []
        # Find cells whose bounding-box collide with the the points
        cell_candidates = geometry.compute_collisions_points(bb_tree, points)
        # Choose one of the cells that contains the point
        colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)
        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        u_values = fenicsx_fct.eval(points_on_proc, cells)
        # here is where we rescale the values of the function with self.exp_radius, to recover the correct values in um
        # we need to do this because we had rescaled all length scales to self.exp_radius in self.set_up_and_solve_finite_element_problem
        # multiply by 1e-6 to get value in m
        return u_values*self.exp_radius*1e-6
    

    def calculate_radial_compliance (self, r_array, D, tip, force, T, k, mesh=None):
        """
        calculate compliance as a fct of distance from the center of the drumhead;
        - r_array: distances from the center at which compliance is to be calculated
        - mesh: - if mesh=None it is assumed that self.prepare_fenicsx_function_space has already been run
                - if a mesh is given, self.prepare_fenicsx_function_space(mesh) is run to create function space variables
        - all other parameters are same as in self.set_up_and_solve_finite_element_problem()
        """
        if mesh is None:
            mesh = self.mesh
        if self.V is None:
            self.prepare_fenicsx_function_space (mesh)
        fenicsx_params = (self.V, self.u, self.f, self.v, self.w, self.x)

        # array in which compliance as a fct of distance is stored
        compl = np.zeros(len(r_array))
        for ii, r in enumerate(r_array):
            point = np.array([0,r], dtype='double')
            self.set_up_and_solve_finite_element_problem (mesh, point, D, tip, force, T, k, fenicsx_params)
            point_3D = np.array([[point[0],point[1],0]], dtype='double')
            # compl[ii] = min(defl_map)/force
            # vals = self.extract_values_at_points (sol, msh, X=[point[0]],Y=[point[1]],Z=[0])
            vals = self.extract_values_at_points (self.current_u_fct, mesh, points=point_3D)
            compl[ii] = vals[0]/force
            self.progress = ii
            # yield self.sleep(1) 
        return (compl)



if __name__ == '__main__':
    # relative_mesh_length = 0.05
    radius = 1

    norm_factor=1.1
    h = 0.015#/norm_factor  #thickness  um.
    E = 0.13 # Young's modulus N/um^2
    nu = 0.26
    D   = E * h**3 / (12 * (1 - nu**2)) #bending rigidity
    tip=4e-4# um
    force = -5e-1  # N
    T= 3e-7 # Tension N/um 
    k=1e13  #shear spring on the edge
    # point = np.array([-0.5, -.2], dtype="double")
    tiff_data  = np.loadtxt('/Users/florian/Documents/Code/afm_youngs_modulus/test/test_forward_calc_mesh_128.txt')
    # r_array = tiff_data[:,0]
    r_array = np.linspace(0,.99, 10)

    calc = CalcCompliance(exp_radius=norm_factor)
    msh = calc.generate_disc_msh(radius, msh_length=0.03)
    calc.prepare_fenicsx_function_space(calc.mesh)
    # compl = calc.calculate_radial_compliance(r_array, h/norm_factor, E*norm_factor**2, nu, tip, force, T*norm_factor, k) * norm_factor
    # compl = calc.calculate_radial_compliance(r_array, h, E/norm_factor**2, nu, tip, force, T/norm_factor, k)
    compl = calc.calculate_radial_compliance(r_array, h, E, nu, tip, force, T, k)
    print(r_array, h, E, nu, tip, force, T, k)

    r_array2 = np.linspace(0,.99*norm_factor, 10)
    calc = CalcCompliance(exp_radius=1)
    msh = calc.generate_disc_msh(norm_factor, msh_length=0.03)
    calc.prepare_fenicsx_function_space(calc.mesh)
    # compl2 = calc.calculate_radial_compliance(r_array2, h, E, nu, tip, force, T, k)
    compl2 = calc.calculate_radial_compliance(r_array2, h, E/norm_factor**2, nu, tip*norm_factor, force, T/norm_factor, k)
    # msh   = calc.mesh
    # fenicsx_params = (calc.V, calc.u, calc.f, calc.v, calc.w, calc.x)
    # calc.set_up_and_solve_finite_element_problem (msh, np.array([0.8,0]), h, E, nu, tip, force, T, k, fenicsx_params)
    # cells, types, x, val_map = calc.get_colormap_plot_data_pyvista(calc.Vproj, fenicsx_fct=calc.current_u_fct)
       
    # grid = pyvista.UnstructuredGrid(cells, types, x)
    # complianceMapPlot = pyvista.Plotter()
    # if val_map is not None:
    #     val_map = val_map/np.max(np.abs(val_map))
    #     grid.point_data["u"] = val_map
    #     grid.set_active_scalars("u")

    #     complianceMapPlot.add_mesh(grid, show_edges=True)
    #     warped = grid.warp_by_scalar()
    #     complianceMapPlot.add_mesh(warped)
    # complianceMapPlot.show()


    # compl = calc.calculate_compliance_map (r_array, msh, h, E, nu, tip, force, T, k, norm_factor)

    plt.figure()

    # plt.scatter(r_array, compl, label=f'Florian min')
    plt.scatter(r_array, compl, label=f'w/ norm factor')
    plt.scatter(r_array2/norm_factor, compl2, label=f'w/o norm factor')
    plt.scatter(tiff_data[:,0], tiff_data[:,1], label='Tiffany')

    plt.legend()
    plt.show()

    #################################################################
    #################################################################
    # plot msh
    #################################################################
    #################################################################
    # cells, types, x = plot.vtk_mesh(V_u)
    # grid = pyvista.UnstructuredGrid(cells, types, x)

    # dat = u_proj.x.array.real
    # dat = dat/max(abs(dat))

    # grid.point_data["u"] = dat
    # grid.set_active_scalars("u")
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(grid, show_edges=True)
    # warped = grid.warp_by_scalar()
    # plotter.add_mesh(warped)
    # plotter.show()






    # #################################################################
    # #################################################################
    # # plot mesh
    # #################################################################
    # #################################################################
    # Visualization with PyVista
    # Extract connectivity (topology) and geometry (points)
    # topology = msh.topology.connectivity(msh.topology.dim, 0)
    # geometry = msh.geometry.x

    # # Convert AdjacencyList topology to NumPy arrays
    # connectivity = []
    # for i in range(len(topology)):
    #     connectivity.append(topology.links(i))
    # connectivity = np.array(connectivity, dtype=np.int32)

    # # Create a PyVista mesh
    # cells = np.hstack([[len(cell)] + cell.tolist() for cell in connectivity])
    # cell_types = np.full(len(connectivity), pyvista.CellType.TRIANGLE, dtype=np.uint8)
    # vtk_mesh = pyvista.UnstructuredGrid(cells, cell_types, geometry)

    # # Plot the mesh
    # vtk_mesh.plot(show_edges=True)