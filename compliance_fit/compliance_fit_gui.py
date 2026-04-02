from PyQt5.QtWidgets import (QApplication,
                             QMainWindow,
                             QPushButton,
                             QHBoxLayout,
                             QVBoxLayout,
                             QWidget,
                             QFileDialog,
                             QTableWidget,
                             QTableWidgetItem,
                             QGraphicsPathItem)
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5 import uic, QtCore
from PyQt5.QtGui import QIcon, QColor, QPainterPath
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path
import numpy as np
import platform
from compliance_fit.calc_compliance_gmsh import CalcCompliance
# from calc_compliance_gmsh import CalcCompliance
import pyvista as pv
from pyvistaqt import QtInteractor
import pyvistaqt
import sys
import threading
from lmfit import minimize, Parameters, fit_report


class ComplianceFitGUI (QMainWindow):
    autoUpdateWidgetsSignal = pyqtSignal(int)
    def __init__(self, operating_system=None):
        super(ComplianceFitGUI, self).__init__()
        
        # different folders in file paths are separated by different symbols depending on the operating software
        # which one it is is important to know when creating correct paths to load/save data
        if operating_system is None:
            operating_system = platform.system()
        if operating_system in ['windows', 'Windows']:
            self.separator = '\\'
        elif operating_system in ['mac', 'Mac', 'Darwin', 'darwin']:
            self.separator = '/'
        else:
            print('operating system not any of the possible options')
            print('current operating system is: ', operating_system)
       
        # import ui file
        path     = str( Path(__file__).absolute() )
        temp     = path.split(self.separator)
        temp[-1] = 'compliance_fit_gui.ui'
        temp_ui     = self.separator.join(temp)
        uic.loadUi(temp_ui, self)

        # serious of fcts to initialize gui variables, plots, widgets, etc
        self.create_compliance_map_plot()
        self.create_radial_compliance_plot()
        self.calcCompliance = CalcCompliance()
        self.initialize_params_variables_and_widgets()
        self.update_params()
        self.update_mesh()
        self.update_compliance_map_plot()
        self.initialize_radial_fit_widgets_and_variables()
        self.initialize_single_compl_map_variables_and_widgets()

    def initialize_single_compl_map_variables_and_widgets (self): 
        self.calcComplMapButton.clicked.connect(self.calc_compl_button_clicked)

        self.xCoordLine.textEdited.connect(self.load_point_changed)
        self.yCoordLine.textEdited.connect(self.load_point_changed)

        self.viewXYButton.clicked.connect(lambda: self.adjust_compliance_map_plot_view(view='xy'))
        self.viewXZButton.clicked.connect(lambda: self.adjust_compliance_map_plot_view(view='xz'))
        self.viewYZButton.clicked.connect(lambda: self.adjust_compliance_map_plot_view(view='yz'))

    def initialize_radial_fit_widgets_and_variables (self):
        self.loadDataButton.clicked.connect(self.load_data_button_clicked)
        self.fit_data   = None
        self.fit_output = None
        self.radial_simulation = None
        self.fileDialog = QFileDialog()
        self.directory  = ''
        self.num_points = int(float(self.numPointsLine.text().strip()))
        self.numPointsLine.setText(str(self.num_points))
        # self.forwardCalcProgressBar.setMaximum(self.num_points)
        # self.forwardCalcProgressBar.setValue(0)
        self.calcRadialComplianceButton.clicked.connect(self.calc_radial_compliance_button_clicked)
        # self.progressBarTimer = QTimer(self)
        # self.progressBarTimer.timeout.connect(self.update_progress_bar)
        self.fitDataButton.clicked.connect(self.fit_data_button_clicked)
        self.saveRadialSimButton.clicked.connect(self.save_radial_sim_button_clicked)
            
        # the fit of the compliance to get Young's Modulus etc will be performed later using lmfit
        # lmfit uses lmfit.Parameters() to handle all possible fit parameters
        self.params = Parameters()
        self.params.add('bending_rigidity', value=self.bending_rigidity, vary=True)
        self.params.add('tip_size', value=self.tip_size, vary=False)
        self.params.add('force', value=self.force, vary=False)
        self.params.add('tension', value=self.tension, vary=True)
        self.params.add('edge_spring_const', value=self.edge_spring_const, vary=False)
    
    def load_point_changed (self):
        # if point where load is applied is changed; button changes to red to indicate
        # inconsistency between plot and variables
        self.calcComplMapButton.setStyleSheet("QPushButton#calcComplMapButton {color: rgb(255, 0, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(255, 0, 0);border-radius: 5px}")
    
    def initialize_params_variables_and_widgets (self):
        self.updateParamsButton.setStyleSheet("QPushButton#updateParamsButton {color: rgb(0, 255, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(0, 255, 0);border-radius: 5px}")
        self.updateMeshButton.setStyleSheet("QPushButton#updateMeshButton {color: rgb(0, 255, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(0, 255, 0);border-radius: 5px}")
        self.updateParamsButton.clicked.connect(self.update_params_button_clicked)
        self.updateMeshButton.clicked.connect(self.update_mesh_button_clicked)

        self.radiusEdit.textEdited.connect(self.mesh_variables_changed)
        self.meshLengthEdit.textEdited.connect(self.mesh_variables_changed)

        self.expRadiusEdit.textEdited.connect(self.params_variables_changed)
        self.bendingRigidityEdit.textEdited.connect(self.params_variables_changed)
        self.tensionEdit.textEdited.connect(self.params_variables_changed)
        self.tipSizeEdit.textEdited.connect(self.params_variables_changed)
        self.forceEdit.textEdited.connect(self.params_variables_changed)
        self.edgeSpringConstEdit.textEdited.connect(self.params_variables_changed)

    def create_radial_compliance_plot(self):
        self.radialCompliancePlot.showAxis('top', show=True)
        self.radialCompliancePlot.showAxis('right', show=True)
        self.radialCompliancePlot.getAxis('top').setStyle(showValues=False)
        self.radialCompliancePlot.getAxis('right').setStyle(showValues=True)
        self.radialCompliancePlot.setTitle('Compliance Radial', **{'color': '#FFF', 'size': '18pt'})
        self.radialCompliancePlot.setLabel('left', 'compliance', units='m/N', **{'color': '#FFF', 'font-size': '14pt'})
        self.radialCompliancePlot.setLabel('bottom', 'r_0/r', units='', **{'color': '#FFF', 'font-size': '14pt'})
        legend_item = self.radialCompliancePlot.addLegend(frame=False, labelTextColor='w', labelTextSize='16pt', offset=(-10,10))

        self.rad_comp_data_plot = pg.ScatterPlotItem([],[], symbol='o')
        color = 'white'
        self.rad_comp_data_plot.setPen(pg.mkPen(color))
        self.rad_comp_data_plot.setBrush(pg.mkBrush(color))
        self.radialCompliancePlot.addItem(self.rad_comp_data_plot)
        legend_item.addItem(self.rad_comp_data_plot, name='data')

        self.rad_comp_fit_plot = pg.ScatterPlotItem([],[], symbol='o')
        color = 'orange'
        self.rad_comp_fit_plot.setPen(pg.mkPen(color))
        self.rad_comp_fit_plot.setBrush(pg.mkBrush(color))
        self.radialCompliancePlot.addItem(self.rad_comp_fit_plot)
        legend_item.addItem(self.rad_comp_fit_plot, name='simulation')
        
    def params_variables_changed (self):
        # if any of the parameters are changed; indicate with red button
        # also disable calcComplMapButton until variables are updated
        self.updateParamsButton.setStyleSheet("QPushButton#updateParamsButton {color: rgb(255, 0, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(255, 0, 0);border-radius: 5px}")
        # self.calcComplMapButton.setStyleSheet("QPushButton#calcComplMapButton {color: rgb(255, 0, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(255, 0, 0);border-radius: 5px}")
        self.calcComplMapButton.setEnabled(False)
    
    def mesh_variables_changed (self):
        # if any of the mesh parameters are changed; indicate with red button
        # also disable calcComplMapButton until variables are updated
        self.updateMeshButton.setStyleSheet("QPushButton#updateMeshButton {color: rgb(255, 0, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(255, 0, 0);border-radius: 5px}")
        # self.calcComplMapButton.setStyleSheet("QPushButton#calcComplMapButton {color: rgb(255, 0, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(255, 0, 0);border-radius: 5px}")
        self.calcComplMapButton.setEnabled(False)
    
    def update_params (self):
        # all length units are in um
        # all other units are in SI units
        # lengths are not rescaled to the experimental radius in this part of the code!!!
        #      -> that happens in calc_compliance_gmsh.py
        self.exp_radius = float(self.expRadiusEdit.text())
        self.calcCompliance.exp_radius = self.exp_radius
        # self.thickness = float(self.thicknessEdit.text())/1e3 # in um
        ################################################################################################################################################
        ################################################################################################################################################
        ################################################################################################################################################
        ################################################################################################################################################
        ################################################################################################################################################
        ################################################################################################################################################
        ################################################################################################################################################
        ################################################################################################################################################
        ################################################################################################################################################
        ################################################################################################################################################
        ################################################################################################################################################
        ################################################################################################################################################
        ################################################################################################################################################
        ################################################################################################################################################
        # need to edit this to get it to units of um
        self.bending_rigidity = float(self.bendingRigidityEdit.text())/1e6 # N*um 
        # self.youngs_modulus = float(self.youngsModulusEdit.text())/1e3 # N/umˆ2
        self.tension = float(self.tensionEdit.text())/1e6 # N/um
        # self.poisson_ratio = float(self.poissonRatioEdit.text())
        self.tip_size = float(self.tipSizeEdit.text())/1e3 # um
        self.force = float(self.forceEdit.text())
        self.edge_spring_const = float(self.edgeSpringConstEdit.text())*1e13
    
    def update_mesh (self):
        self.radius = float(self.radiusEdit.text())
        self.mesh_length = float(self.meshLengthEdit.text())
        self.calcCompliance.current_u_fct = None
        self.calcCompliance.generate_disc_msh(radius=self.radius, msh_length=self.mesh_length)
        self.calcCompliance.prepare_fenicsx_function_space (self.calcCompliance.mesh)

    def update_params_button_clicked (self):
        try:
            self.update_params()
            self.calcCompliance.current_u_fct = None
            self.update_compliance_map_plot()
            self.updateParamsButton.setStyleSheet("QPushButton#updateParamsButton {color: rgb(0, 255, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(0, 255, 0);border-radius: 5px}")
            self.calcComplMapButton.setEnabled(True)
            # self.calcComplMapButton.setStyleSheet("QPushButton#calcComplMapButton {color: rgb(0, 255, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(0, 255, 0);border-radius: 5px}")
            self.update_radial_compliance_plot(simulation=np.array([[[np.nan],[np.nan]]]))
            
            self.params['bending_rigidity'].value  = self.bending_rigidity
            self.params['tip_size'].value          = self.tip_size
            self.params['force'].value             = self.force
            self.params['tension'].value           = self.tension
            self.params['edge_spring_const'].value = self.edge_spring_const

        except Exception as e:
            self.updateParamsButton.setStyleSheet("QPushButton#updateParamsButton {color: rgb(255, 0, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(255, 0, 0);border-radius: 5px}")
            # self.calcComplMapButton.setStyleSheet("QPushButton#calcComplMapButton {color: rgb(255, 0, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(255, 0, 0);border-radius: 5px}")
            print("Error updating parameters and plot ...")
            print(e)
    
    def update_mesh_button_clicked (self):
        try:
            self.update_mesh()
            self.update_compliance_map_plot()
            self.updateMeshButton.setStyleSheet("QPushButton#updateMeshButton {color: rgb(0, 255, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(0, 255, 0);border-radius: 5px}")
            self.calcComplMapButton.setEnabled(True)
            # self.calcComplMapButton.setStyleSheet("QPushButton#calcComplMapButton {color: rgb(0, 255, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(0, 255, 0);border-radius: 5px}")
            self.update_radial_compliance_plot(simulation=np.array([[[np.nan],[np.nan]]]))
        except Exception as e:
            self.updateMeshButton.setStyleSheet("QPushButton#updateMeshButton {color: rgb(255, 0, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(255, 0, 0);border-radius: 5px}")
            # self.calcComplMapButton.setStyleSheet("QPushButton#calcComplMapButton {color: rgb(255, 0, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(255, 0, 0);border-radius: 5px}")
            print("Error updating mesh params and plot ...")
            print(e)
    
    def create_compliance_map_plot (self):
        self.complianceMapPlot = QtInteractor(self.complianceMapWidget)
        self.complianceMapPlot.setGeometry(20, 12, 550, 400)
        
        self.complianceMapPlot.view_xy()
        self.complianceMapPlot.add_axes(line_width=8,cone_radius=0.5,shaft_length=0.6,tip_length=0.4,ambient=0.5,label_size=(0.5, 0.2))
        self.complianceMapPlot.reset_camera()
        return 1
    
    def adjust_compliance_map_plot_view (self, view):
        if view=='xy':
            self.complianceMapPlot.view_xy()
        elif view=='xz':
            self.complianceMapPlot.view_xz()
        elif view=='yz':
            self.complianceMapPlot.view_yz()
        self.complianceMapPlot.reset_camera()

    def update_compliance_map_plot (self):
        cells, types, x, val_map = self.calcCompliance.get_colormap_plot_data_pyvista(self.calcCompliance.Vproj, fenicsx_fct=self.calcCompliance.current_u_fct)
        self.complianceMapPlot.clear()
        self.grid = None # not sure if this is really necessary, but I think pv.UnstructuredGrid is qutie large in memory, so I'm clearing this variable here to free up some memory
        self.grid = pv.UnstructuredGrid(cells, types, x)
        if val_map is not None:
            val_map = val_map/np.max(np.abs(val_map))#*self.radius
            self.grid.point_data["u"] = val_map
            self.grid.set_active_scalars("u")
    
            self.complianceMapPlot.add_mesh(self.grid, show_edges=True, show_scalar_bar=False)
            self.warped = self.grid.warp_by_scalar()
            self.complianceMapPlot.add_mesh(self.warped, show_scalar_bar=False)
        else:
            self.complianceMapPlot.add_mesh(self.grid, show_edges=True, show_scalar_bar=False)

        self.complianceMapPlot.reset_camera()
        return 1

    def calc_compl_button_clicked (self):
        try:
            x_coord = float(self.xCoordLine.text())
            y_coord = float(self.yCoordLine.text())
            point   = np.array([x_coord, y_coord])
            
            fenicsx_params = (self.calcCompliance.V, self.calcCompliance.u, self.calcCompliance.f, self.calcCompliance.v, self.calcCompliance.w, self.calcCompliance.x)            
            D     = self.bending_rigidity
            tip   = self.tip_size
            force = self.force
            T     = self.tension
            k     = self.edge_spring_const
            
            self.calcCompliance.set_up_and_solve_finite_element_problem (self.calcCompliance.mesh, point, D, tip, force, T, k, fenicsx_params)
            self.calcCompliance.get_colormap_plot_data_pyvista(self.calcCompliance.Vproj, fenicsx_fct=self.calcCompliance.current_u_fct)
            self.update_compliance_map_plot()

            self.calcComplMapButton.setStyleSheet("QPushButton#calcComplMapButton {color: rgb(0, 255, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(0, 255, 0);border-radius: 5px}")
        except Exception as e:
            print('Error calculating compliance map ...')
            print(e)

    def load_data_button_clicked (self):
        try:
            options = QFileDialog.Options()
            filename, _ = self.fileDialog.getOpenFileName(self, "Open Fit Data",self.directory,"All Files (*)", options=options)
            self.directory = self.separator.join(filename.split(self.separator)[:-1])
            # self.fit_data = np.loadtxt(filename, delimiter=',')
            data = np.load(filename)
            self.fit_data = data['radial compliance fit data (r/r0 and m/N)']
            self.expRadiusEdit.setText(str(np.round(data['radius (um)'],3)))
            self.update_params()
            self.update_radial_compliance_plot(data=self.fit_data)
        except Exception as e:
            print('Error loading fit data ...')
            print(e)

    def update_radial_compliance_plot (self, data=None, simulation=None):
        try:
            if data is not None:
                # self.rad_comp_data_plot.setData(data[:,0], data[:,1])
                self.rad_comp_data_plot.setData(data[0], data[1])
            if simulation is not None:
                self.rad_comp_fit_plot.setData(simulation[:,0], simulation[:,1])
            self.radialCompliancePlot.getViewBox().autoRange()
        except Exception as e:
            print('Error updating radial compliance plot ...')
            print(e)
    
    def calc_radial_compliance (self):
        try:
            self.num_points = int(self.numPointsLine.text())
            self.numPointsLine.setText(str(self.num_points))
            # self.forwardCalcProgressBar.setMaximum(self.num_points)
            # self.forwardCalcProgressBar.setValue(0)

            self.calcCompliance.exp_radius = self.exp_radius
            r_array    = np.linspace(0,0.99*self.radius, self.num_points)
            # self.progressBarTimer.start(500) # check every 500 milliseconds
            compliance = self.calcCompliance.calculate_radial_compliance(r_array, self.bending_rigidity, self.tip_size, self.force, self.tension, self.edge_spring_const)
            # self.progressBarTimer.stop()
            self.radial_simulation = np.array([r_array/self.radius, compliance]).T
            self.update_radial_compliance_plot(simulation=self.radial_simulation)
            self.calcRadialComplianceButton.setEnabled(True)
            self.calcRadialComplianceButton.setStyleSheet("QPushButton#calcRadialComplianceButton {color: rgb(0, 208, 172);background-color:rgb(255, 255, 255);border: 2px solid rgb(0, 0, 0);border-radius: 0px}")
            self.calcRadialComplianceButton.setText("Calculate\nradial\ncompliance")
        except Exception as e:
            print('Error calculating radial compliance ...')
            print(e)
        
    def calc_radial_compliance_button_clicked (self):
        self.calcRadialComplianceButton.setEnabled(False)
        self.calcRadialComplianceButton.setStyleSheet("QPushButton#calcRadialComplianceButton {color: rgb(181,181,181);background-color:rgb(255, 255, 255);border: 2px solid rgb(181,181,181);border-radius: 5px}")
        self.calcRadialComplianceButton.setText("Calculation\nrunning\n...")

        calc_thread = threading.Thread(target=self.calc_radial_compliance)
        calc_thread.start()

               
    # def update_progress_bar (self):
    #     progress = self.calcCompliance.progress
    #     self.forwardCalcProgressBar.setValue(progress+1)

    def residual_function (self, params):
        # r_array, compl_data = self.fit_data[:,0], self.fit_data[:,1]
        r_array, compl_data = self.fit_data[0], self.fit_data[1]

        self.bending_rigidity = params['bending_rigidity'].value
        self.tip_size = params['tip_size'].value
        self.force = params['force'].value
        self.tension = params['tension'].value
        self.edge_spring_const = params['edge_spring_const'].value

        compl_sim = self.calcCompliance.calculate_radial_compliance(r_array*self.radius, self.bending_rigidity, self.tip_size, self.force, self.tension, self.edge_spring_const)
        
        self.radial_simulation = np.array([r_array, compl_sim]).T
        self.update_radial_compliance_plot(simulation=self.radial_simulation)

        self.bendingRigidityEdit.setText(str(np.round(self.bending_rigidity*1e6,5)))
        self.tipSizeEdit.setText(str(self.tip_size*1e3))
        self.forceEdit.setText(str(self.force))
        self.tensionEdit.setText(str(np.round(self.tension*1e6,5)))
        self.edgeSpringConstEdit.setText(str(self.edge_spring_const/1e13))
        
        return compl_sim-compl_data
    
    def run_fit (self):
        self.fit_output=None
        try:
            if self.fit_data is not None:
                fit_output = minimize(self.residual_function, self.params, method='leastsq')
                self.fit_output = fit_output
            else:
                print('you need to load data first before you can run the fit')

        except Exception as e:
            print('Error fitting ...')
            print(e)
        self.fitDataButton.setEnabled(True)
        self.fitDataButton.setStyleSheet("QPushButton#fitDataButton {color: rgb(255, 125, 122);background-color:rgb(255, 255, 255);border: 0px solid rgb(0, 0, 0);border-radius: 0px}")
        self.fitDataButton.setText('Fit Data')
        return self.fit_output
    
    def fit_data_button_clicked (self):
        # delete the current compliance map plot
        # this is to avoid confusion as to whether the displayed compliance map corresponds to the shown parameters
        self.calcCompliance.current_u_fct = None
        self.update_compliance_map_plot()
        
        self.fitDataButton.setEnabled(False)
        self.fitDataButton.setStyleSheet("QPushButton#fitDataButton {color: rgb(181,181,181);background-color:rgb(255, 255, 255);border: 0px solid rgb(0, 0, 0);border-radius: 0px}")
        self.fitDataButton.setText('Fit running ...')
        # QApplication.processEvents()
        fit_thread = threading.Thread(target=self.run_fit)
        fit_thread.start()

    def save_radial_sim_button_clicked (self):
        try:
            header1 = 'Calculation Parameters:\n'
            header2 = f'Mesh radius (um) = {self.radiusEdit.text()};\tCharacteristic mesh length = {self.meshLengthEdit.text()};\n'
            header3 = f'Tension (N/m) = {self.tensionEdit.text()};\t Bending Rigidity (N*pm) = {self.bendingRigidityEdit.text()};\n'
            header4 = f'Experimental radius (um) = {self.expRadiusEdit.text()};\n'
            header5 = f'Tip size (nm) = {self.tipSizeEdit.text()};\tForce (N) = {self.forceEdit.text()};\tEdge spring constant (1e13) = {self.edgeSpringConstEdit.text()};\n'
            header6 = '\n'+20*'-'+20*'-'+'\n\n'
            header7 = 'r_0/r, compliance'
            header  = header1+header2+header3+header4+header5+header6+header7

            options = QFileDialog.Options()
            save_name, _ = self.fileDialog.getSaveFileName(self, "Save Fit Result",self.directory,"Data Files (*.dat);;Text Files (*.txt);;All Files (*)", options=options)
            np.savetxt(save_name, self.radial_simulation, delimiter=',', header=header)
        except Exception as e:
            print('Error saving fit ...')
            print(e)
        

    def closeEvent(self, event):
        # Properly clean up the plotter when the window is closed
        self.complianceMapPlot.close()
        self.calcCompliance.current_u_fct = None
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication([])
    win = ComplianceFitGUI()
    win.show()
    app.exec()