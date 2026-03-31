from PyQt5.QtWidgets import (QApplication,
                             QMainWindow,
                             QPushButton,
                             QHBoxLayout,
                             QVBoxLayout,
                             QWidget,
                             QFileDialog,
                             QTableWidget,
                             QTableWidgetItem,
                             QGraphicsEllipseItem)
from PyQt5.QtCore import QTimer, pyqtSignal, QRectF
from PyQt5 import uic, QtCore
from PyQt5.QtGui import QIcon, QColor
import pyqtgraph as pg
from pathlib import Path
import numpy as np
import threading
from time import time
import os
import sys
from copy import deepcopy
import ctypes
import json
from numpy.polynomial import polynomial
import platform
from data_processor.afm_data_processor import AFMForceMapData

class DataProcessorGUI (QMainWindow):
    # the signal needs to be defined as a "class-level" attribute, so here, outside of __init__
    autoUpdateWidgetsSignal = pyqtSignal(int)
    def __init__(self,operating_system=None):
        super(DataProcessorGUI, self).__init__()
        
        # different folders in file paths are separated by different symbols depending on the operating software
        # which one it is is important to know when creating correct paths to load/save data
        if operating_system is None:
            operating_system = platform.system()
        if operating_system in ['windows', 'Windows']:
            self.separator = '\\'
            icon_name = 'window_icon.png'
        elif operating_system in ['mac', 'Mac', 'Darwin', 'darwin']:
            self.separator = '/'
            icon_name = 'window_icon.icns'
        else:
            print('operating system not any of the possible options')
            print('current operating system is: ', operating_system)
            sys.exit()
        
        # import ui file
        path     = str( Path(__file__).absolute() )
        temp     = path.split(self.separator)
        temp[-1] = 'afm_data_processor_gui.ui'
        temp_ui     = self.separator.join(temp)
        uic.loadUi(temp_ui, self)
        
        temp[-1]  = icon_name
        temp_icon = self.separator.join(temp)
        self.setWindowIcon(QIcon(temp_icon))

        self.initialize_data_and_fit_variables()
        self.initialize_directory_widgets()
        self.create_force_curve_plot()
        self.initialize_fit_widgets()
        self.create_compliance_map_plot()
        self.create_compliance_radial_plot()
        self.initialize_compliance_map_widgets()
        self.initialize_radial_compliance_widgets()
        
        self.afmForceMapData = AFMForceMapData()
        self.active_compliance_map = None
        self.active_radial_compliance_data = None
        self.active_radial_compliance_fit_data = None
        self.radius = 0
        self.radius_um = np.nan
        self.scan_win_size = 0
    
    def initialize_directory_widgets(self):
        # Widgets and Variables for loading files
        self.current_file_index = 0
        self.fileDialog = QFileDialog()
        self.browseButton.clicked.connect(self.browse_button_clicked)
        self.plotterDir = None
        self.fileList.doubleClicked.connect(lambda: self.fileListDoubleClicked(rescale_axes=True))

    def initialize_data_and_fit_variables(self):
        # variables to store all data
        self.filenames     = None
        self.current_force_curve = None

        # variables to store fit results
        self.single_fit_result = None # fit results of a single fit
        self.fit_results_list = [] # contains fit results for all AFM force curves

    def initialize_fit_widgets (self):
        self.FitComplianceButton.setStyleSheet("QPushButton#FitComplianceButton {color: rgb(0, 255, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(0, 255, 0);border-radius: 5px}")
        self.FitComplianceButton.clicked.connect(self.fit_compliance_button_clicked)
        self.kTiplineEdit.textEdited.connect(self.k_tip_changed)
    
    def create_force_curve_plot (self):
        legend_item = self.individualForcePlot.addLegend(frame=False, labelTextColor='w', labelTextSize='14pt')

        self.approach_data_plot = pg.PlotCurveItem([],[])
        self.approach_data_plot.setPen(pg.mkPen(color='g', width=3, style=pg.QtCore.Qt.DashLine))
        self.individualForcePlot.addItem(self.approach_data_plot)
        legend_item.addItem(self.approach_data_plot, name='approach')

        self.approach_fit_data_plot = pg.PlotCurveItem([],[])
        self.approach_fit_data_plot.setPen(pg.mkPen(color='g', width=5))
        self.individualForcePlot.addItem(self.approach_fit_data_plot)

        self.approach_fit_plot = pg.PlotCurveItem([],[])
        self.approach_fit_plot.setPen(pg.mkPen(color='white', width=3))
        self.individualForcePlot.addItem(self.approach_fit_plot)

        self.retract_data_plot = pg.PlotCurveItem([],[])
        self.retract_data_plot.setPen(pg.mkPen(color='red', width=3, style=pg.QtCore.Qt.DashLine))
        self.individualForcePlot.addItem(self.retract_data_plot)
        legend_item.addItem(self.retract_data_plot, name='retract')

        self.retract_fit_data_plot = pg.PlotCurveItem([],[])
        self.retract_fit_data_plot.setPen(pg.mkPen(color='red', width=5))
        self.individualForcePlot.addItem(self.retract_fit_data_plot)

        self.retract_fit_plot = pg.PlotCurveItem([],[])
        self.retract_fit_plot.setPen(pg.mkPen(color='white', width=3))
        self.individualForcePlot.addItem(self.retract_fit_plot)

        self.individualForcePlot.showAxis('top', show=True)
        self.individualForcePlot.showAxis('right', show=True)
        self.individualForcePlot.getAxis('top').setStyle(showValues=False)
        self.individualForcePlot.getAxis('right').setStyle(showValues=False)
        self.individualForcePlot.setLabel('left', 'AFM Tip Deflection', units='m', **{'color': '#FFF', 'font-size': '12pt'})
        self.individualForcePlot.setLabel('bottom', 'Z-Sensor Distance', units='m', **{'color': '#FFF', 'font-size': '12pt'})

    def create_compliance_map_plot (self):
        self.complianceMapPlot.showAxis('top', show=True)
        self.complianceMapPlot.showAxis('right', show=True)
        self.complianceMapPlot.getAxis('top').setStyle(showValues=False)
        self.complianceMapPlot.getAxis('right').setStyle(showValues=False)
        self.complianceMapPlot.setTitle('Compliance Map', **{'color': '#FFF', 'size': '18pt'})    
        self.complianceMapPlot.setAspectLocked(lock=True, ratio=1)

        self.compliancemap_imageItem = pg.ImageItem()
        colorMap = pg.colormap.get('viridis')
        self.compliancemap_imageItem.setLookupTable(colorMap.getLookupTable())
        self.complianceMapPlot.addItem(self.compliancemap_imageItem)

        self.circle_center = np.array([0,0])
        self.circle_center_plot = pg.TargetItem(pos=(self.circle_center[0], self.circle_center[1]), size=10, symbol='o', pen=pg.mkPen('r'), brush=pg.mkBrush('r'), movable=True)
        self.complianceMapPlot.addItem(self.circle_center_plot)

        self.radius_point = np.array([0,0.1])
        self.radius = np.sqrt((self.circle_center[0]-self.radius_point[0])**2 + (self.circle_center[1]-self.radius_point[1])**2)
        self.radius_point_plot = pg.TargetItem(pos=(self.radius_point[0], self.radius_point[1]), size=10, symbol='o', pen=pg.mkPen('r'), brush=pg.mkBrush('r'), movable=True)
        self.complianceMapPlot.addItem(self.radius_point_plot)
       
        self.circle = QGraphicsEllipseItem(self.circle_center[0] - self.radius, self.circle_center[1] - self.radius, 2*self.radius, 2*self.radius)
        self.circle.setPen(pg.mkPen(color='r', width=2))  # Set the pen color and width
        self.complianceMapPlot.addItem(self.circle)

        try:
            num_pixels = np.shape(self.afmForceMapData.raw_compliance_array)[0]
            self.scan_win_size = float(self.scanWinSizelineEdit.text())
            self.radius_um = self.radius * self.scan_win_size / num_pixels
            self.radius_um_lineEdit.setText(f'{self.radius_um:.2f}')
        except:
            self.radius_um_lineEdit.setText('nan')
            self.radius_um = np.nan
    
    def create_compliance_radial_plot (self):
        self.complianceRadialPlot.showAxis('top', show=True)
        self.complianceRadialPlot.showAxis('right', show=True)
        self.complianceRadialPlot.getAxis('top').setStyle(showValues=False)
        self.complianceRadialPlot.getAxis('right').setStyle(showValues=False)
        self.complianceRadialPlot.setTitle('Compliance Radial', **{'color': '#FFF', 'size': '18pt'})
        self.complianceRadialPlot.setLabel('left', 'compliance', units='m/N', **{'color': '#FFF', 'font-size': '12pt'})
        self.complianceRadialPlot.setLabel('bottom', 'r_0/r', units='', **{'color': '#FFF', 'font-size': '12pt'})

        self.rad_comp_plot = pg.ScatterPlotItem([],[], symbol='o')
        color = 'white'
        self.rad_comp_plot.setPen(pg.mkPen(color))
        self.rad_comp_plot.setBrush(pg.mkBrush(color))
        self.complianceRadialPlot.addItem(self.rad_comp_plot)

        self.fit_rad_comp_plot = pg.ScatterPlotItem([],[], symbol='o')
        color = 'orange'
        self.fit_rad_comp_plot.setPen(pg.mkPen(color))
        self.fit_rad_comp_plot.setBrush(pg.mkBrush(color))
        self.complianceRadialPlot.addItem(self.fit_rad_comp_plot)
    
    def initialize_compliance_map_widgets (self):    
        self.circle_center_plot.sigPositionChanged.connect(self.update_circle_params_center)
        self.radius_point_plot.sigPositionChanged.connect(self.update_circle_params_radius)
        self.smoothComplianceButton.clicked.connect(self.smooth_compliance_map_data)
        self.findCenterButton.clicked.connect(self.find_circle_center)
    
    def initialize_radial_compliance_widgets(self):
        self.projectRadialComplianceButton.clicked.connect(self.project_radial_compliance_button_clicked)
        self.createRadCompFitDataButton.clicked.connect(self.create_rad_com_fit_data_button_clicked)
        # self.saveFitDataButton.clicked.connect(self.save_fit_data_button_clicked)
        # self.saveRadialDataButton.clicked.connect(self.save_radial_data_button_clicked)
        self.saveAllButton.clicked.connect(self.save_all_button_clicked)
            
    def browse_button_clicked (self, plot_dir=None):
        self.fileList.clear()
        self.all_results = None
        try:
            if plot_dir is None or not plot_dir:
                plot_dir = self.fileDialog.getExistingDirectory(self, "Select Directory")
            if plot_dir:
                self.plotterDir = plot_dir
                self.direcotry_print.setText(self.plotterDir)
                x_name = self.xnamelineEdit.text().strip()
                y_name = self.ynamelineEdit.text().strip()
                self.afmForceMapData.get_filenames(self.plotterDir, x_name, y_name)
                for index in self.afmForceMapData.x_index:
                    self.fileList.addItem(f'{index[0]}\t{index[1]}')
                self.afmForceMapData.load_data()
            self.afmForceMapData.approach_fit_data, self.afmForceMapData.retract_fit_data = [], []
            self.afmForceMapData.approach_fit, self.afmForceMapData.retract_fit = [], []     

        except Exception as e:
            print("Error loading data ...")
            print(e)

    def fileListDoubleClicked(self, rescale_axes=False):
        self.current_file_index = self.fileList.currentRow()
        current_item = self.fileList.currentItem().text().split()
        title = f'Row: {current_item[0]}; Column: {current_item[1]}'        
        self.updateForcePlot(self.current_file_index, title, rescale_axes=rescale_axes)

    def updateForcePlot (self, current_file_index=None, title=None, rescale_axes=False):
        if current_file_index is None:
            current_file_index = self.current_file_index
        if not title is None:
            self.individualForcePlot.setTitle(title, **{'color': '#FFF', 'size': '15pt'})
        
        approach_data = self.afmForceMapData.approach_data[self.current_file_index]
        retract_data = self.afmForceMapData.retract_data[self.current_file_index]
        self.approach_data_plot.setData(approach_data[0], approach_data[1])
        self.retract_data_plot.setData(retract_data[0], retract_data[1])

        if len(self.afmForceMapData.approach_fit)==len(self.afmForceMapData.approach_data):
            self.approach_fit_data_plot.setData(self.afmForceMapData.approach_fit_data[self.current_file_index][0], self.afmForceMapData.approach_fit_data[self.current_file_index][1])
            self.approach_fit_plot.setData(self.afmForceMapData.approach_fit[self.current_file_index][0], self.afmForceMapData.approach_fit[self.current_file_index][1])
            self.retract_fit_data_plot.setData(self.afmForceMapData.retract_fit_data[self.current_file_index][0], self.afmForceMapData.retract_fit_data[self.current_file_index][1])
            self.retract_fit_plot.setData(self.afmForceMapData.retract_fit[self.current_file_index][0], self.afmForceMapData.retract_fit[self.current_file_index][1])
        else:
            self.approach_fit_data_plot.setData([],[])
            self.approach_fit_plot.setData([],[])
            self.retract_fit_data_plot.setData([],[])
            self.retract_fit_plot.setData([],[])

        if rescale_axes:
            self.individualForcePlot.setXRange(min([min(approach_data[0]), min(retract_data[0])]), max([max(approach_data[0]), max(retract_data[0])]))
            self.individualForcePlot.setYRange(min([min(approach_data[1]), min(retract_data[1])]), max([max(approach_data[1]), max(retract_data[1])]))

    def fit_compliance_button_clicked (self):
        try:
            x_index = self.afmForceMapData.x_index
            approach_data = self.afmForceMapData.approach_data
            retract_data  = self.afmForceMapData.retract_data

            k_tip = float(self.kTiplineEdit.text())
            compliance_map = self.afmForceMapData.fit_map_compliance(x_index, approach_data, retract_data, k_tip=k_tip, fit_type='linear')
            self.updateForcePlot()
            self.update_compliance_map_plot(compliance_map)
            if self.circle_center[0]==0 and self.circle_center[1]==0 and self.radius_point[0]==0 and self.radius_point[1]==0.1:
                self.circle_center_plot.setPos((int(np.shape(compliance_map)[0]/2), int(np.shape(compliance_map)[1]/2)))
                self.update_circle_params_center()
                self.radius_point_plot.setPos((int(np.shape(compliance_map)[0]*3/4), int(np.shape(compliance_map)[1]/2)))
                self.update_circle_params_radius()
            
            self.active_radial_compliance_data = None
            self.active_radial_compliance_fit_data = None
            self.update_radial_compliance_plot(self.active_radial_compliance_data,self.active_radial_compliance_fit_data)
            self.FitComplianceButton.setStyleSheet("QPushButton#FitComplianceButton {color: rgb(0, 255, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(0, 255, 0);border-radius: 5px}")
        
        except Exception as e:
            self.afmForceMapData.approach_fit_data, self.afmForceMapData.retract_fit_data = [], []
            self.afmForceMapData.approach_fit, self.afmForceMapData.retract_fit = [], []
            self.afmForceMapData.raw_compliance_array = None
            print("Error fitting compliance ...")
            print(e)

    def update_compliance_map_plot (self, compliance_map):
        self.compliancemap_imageItem.setImage(compliance_map.T)
        # self.complianceMapPlot.setRange(rect=self.compliancemap_imageItem.boundingRect())
        self.complianceMapPlot.setRange(xRange=(0,np.shape(compliance_map)[0]),yRange=(0,np.shape(compliance_map)[1]), padding=0)
        self.active_compliance_map = compliance_map
    
    def update_circle_params_center (self):
        pos = np.array([self.circle_center_plot.pos().x(),self.circle_center_plot.pos().y()])
        diff = -self.circle_center+pos
        self.circle_center = pos
        self.circleCenterlineEdit.setText(f'{self.circle_center[0]:.2f}\t{self.circle_center[1]:.2f}')

        self.radius_point = self.radius_point+diff
        self.radius_point_plot.setPos(self.radius_point[0],self.radius_point[1])

        self.radius = np.sqrt((self.circle_center[0]-self.radius_point[0])**2 + (self.circle_center[1]-self.radius_point[1])**2)
        self.radiuslineEdit.setText(f'{self.radius:.2f}')
        try:
            num_pixels = np.shape(self.afmForceMapData.raw_compliance_array)[0]
            self.scan_win_size = float(self.scanWinSizelineEdit.text())
            self.radius_um = self.radius * self.scan_win_size / num_pixels
            self.radius_um_lineEdit.setText(f'{self.radius_um:.2f}')
        except:
            self.radius_um_lineEdit.setText('nan')
            self.radius_um = np.nan

        self.circle.setRect(self.circle_center[0] - self.radius, self.circle_center[1] - self.radius, 2*self.radius, 2*self.radius)

    def update_circle_params_radius (self):
        pos = self.radius_point_plot.pos()
        self.radius_point = np.array([pos.x(),pos.y()])
        self.radius = np.sqrt((self.circle_center[0]-self.radius_point[0])**2 + (self.circle_center[1]-self.radius_point[1])**2)
        self.radiuslineEdit.setText(f'{self.radius:.2f}')
        try:
            num_pixels = np.shape(self.afmForceMapData.raw_compliance_array)[0]
            self.scan_win_size = float(self.scanWinSizelineEdit.text())
            self.radius_um = self.radius * self.scan_win_size / num_pixels
            self.radius_um_lineEdit.setText(f'{self.radius_um:.2f}')
        except:
            self.radius_um_lineEdit.setText('nan')
            self.radius_um = np.nan

        self.circle.setRect(self.circle_center[0] - self.radius, self.circle_center[1] - self.radius, 2*self.radius, 2*self.radius)
    
    def smooth_compliance_map_data (self):
        try:
            self.afmForceMapData.post_process_compliance_array(self.afmForceMapData.raw_compliance_array, threshold_compliance=6)
            self.update_compliance_map_plot(self.afmForceMapData.processed_compliance_array)
        
        except Exception as e:
            print('Error creating processed compliance map ...')
            print(e)
    
    def find_circle_center (self):
        try:
            x, y = self.afmForceMapData.find_circle_manual(self.active_compliance_map, self.circle_center)
            self.circle_center_plot.setPos(x,y)
            self.update_circle_params_center()
        except Exception as e:
            print('Could not find center of circle ...')
            print(e)

    def project_radial_compliance_button_clicked (self):
        try:
            x, y = self.circle_center_plot.pos().x(), self.circle_center_plot.pos().y()
            # making the senter of the circle integer pixels
            # not sure if this is super correct, but we definitely don't have more resolution than 1 pixel
            # but not sure if just rounding to the next integer is the correct approach
            # but then again shouldn't matter much
            self.circle_center_plot.setPos(int(np.round(x,0)), int(np.round(y,0)))
            self.update_circle_params_center()

            num_pixels = np.shape(self.active_compliance_map)[0]
            scan_win_size = float(self.scanWinSizelineEdit.text())
            radius_um = self.radius * scan_win_size / num_pixels
            distance, compliance = self.afmForceMapData.create_radial_plot_data(self.active_compliance_map, self.circle_center.astype('int'), scan_win_size, radius_um, zero_compl=0)
            self.active_radial_compliance_data = np.array([distance, compliance])
            self.active_radial_compliance_fit_data = None
            self.update_radial_compliance_plot(self.active_radial_compliance_data,self.active_radial_compliance_fit_data)
        except Exception as e:
            print('Error calculating radial compliance ...')
            print(e)
    
    def update_radial_compliance_plot (self, radial_compliance_data=None, radial_compliance_fit_data=None):
        try:
            if radial_compliance_data is not None:
                self.rad_comp_plot.setData(radial_compliance_data[0], radial_compliance_data[1])
            else:
                self.rad_comp_plot.setData([],[])
            if radial_compliance_fit_data is not None:
                self.fit_rad_comp_plot.setData(radial_compliance_fit_data[0], radial_compliance_fit_data[1])
            else:
                self.fit_rad_comp_plot.setData([],[])
        except Exception as e:
            print('Error updating radial compliance plot ...')
            print(e)
    
    def create_rad_com_fit_data_button_clicked (self):
        try:
            x, y = self.circle_center_plot.pos().x(), self.circle_center_plot.pos().y()
            self.circle_center_plot.setPos(int(x), int(y))
            self.update_circle_params_center()
            circle_center = self.circle_center.astype(int)

            num_pixels = np.shape(self.active_compliance_map)[0]
            scan_win_size = float(self.scanWinSizelineEdit.text())
            radius_um = self.radius * scan_win_size / num_pixels
            rdivs = int(self.numPointsFitDatalineEdit.text())
            distance, compliance, zero_compliance = self.afmForceMapData.create_radial_fit_data (self.active_compliance_map, circle_center, radius_um, scan_win_size, r_divs=rdivs, theta_divs=80, calib_boundary=1.2)
            self.active_radial_compliance_fit_data = np.array([distance, compliance])

            distance, compliance = self.afmForceMapData.create_radial_plot_data(self.active_compliance_map, circle_center, scan_win_size, radius_um, zero_compl=zero_compliance)
            self.active_radial_compliance_data = np.array([distance, compliance])
            
            self.update_radial_compliance_plot(self.active_radial_compliance_data,self.active_radial_compliance_fit_data)
        except Exception as e:
            print('Error creating fit data ...')
            print(e)

    # def save_fit_data_button_clicked (self):
    #     try:
    #         temp = self.plotterDir.split(self.separator)[:-1]
    #         init_directory = self.separator.join(temp)
    #         save_name, _ = self.fileDialog.getSaveFileName(self, "Save Fit Data",init_directory,"Data Files (*.dat);;Text Files (*.txt);;All Files (*)", options=QFileDialog.Options())
    #         header1 = f'circle center (pixels) = {self.circle_center}'
    #         header2 = f'radius (um) = {self.radius_um}'
    #         header3 = f'scan window size (um) = {self.scan_win_size}'
    #         header4 = 'r0/r,compliance'
    #         header  = f'{header1}\n{header2}\n{header3}\n{header4}'
    #         np.savetxt(save_name, self.active_radial_compliance_fit_data.T, delimiter=',', header=header)
    #     except Exception as e:
    #         print('Error saving data ...')
    #         print(e)

    # def save_radial_data_button_clicked (self):
    #     try:
    #         temp = self.plotterDir.split(self.separator)[:-1]
    #         init_directory = self.separator.join(temp)
    #         save_name, _ = self.fileDialog.getSaveFileName(self, "Save Fit Data",init_directory,"Data Files (*.dat);;Text Files (*.txt);;All Files (*)", options=QFileDialog.Options())
    #         header1 = f'circle center (pixels) = {self.circle_center}'
    #         header2 = f'radius (um) = {self.radius_um}'
    #         header3 = f'scan window size (um) = {self.scan_win_size}'
    #         header4 = 'r0/r,compliance'
    #         header  = f'{header1}\n{header2}\n{header3}\n{header4}'
    #         np.savetxt(save_name, self.active_radial_compliance_data.T, delimiter=',', header=header)
    #     except Exception as e:
    #         print('Error saving all data ...')
    #         print(e)
    
    def save_all_button_clicked (self):
        try:
            temp = self.plotterDir.split(self.separator)[:-1]
            init_directory = self.separator.join(temp)
            save_name, _ = self.fileDialog.getSaveFileName(self, "Save Data",init_directory,"npz (*.npz);;All Files (*)", options=QFileDialog.Options())
            if not save_name.endswith(".npz"):
                temp = save_name.split('.')
                if len(temp)>=2:
                    temp[-1] = 'npz'
                    save_name = '.'.join(temp)
                else:
                    save_name = save_name+'.npz'
            save_dict = {}
            save_dict['circle center (pixels)']                    = self.circle_center
            save_dict['radius (um)']                               = self.radius_um
            save_dict['radius (pixel)']                            = self.radius
            save_dict['scan window size (um)']                     = self.scan_win_size
            save_dict['tip spring constant']                       = float(self.kTiplineEdit.text())
            save_dict['radial compliance data (r/r0 and m/N)']     = self.active_radial_compliance_data
            save_dict['radial compliance fit data (r/r0 and m/N)'] = self.active_radial_compliance_fit_data
            save_dict['compliance map (m/N)']                     = self.afmForceMapData.processed_compliance_array
            np.savez_compressed(save_name, **save_dict)
        except Exception as e:
            print('Error saving data ...')
            print(e)
    
    def k_tip_changed (self):
        self.FitComplianceButton.setStyleSheet("QPushButton#FitComplianceButton {color: rgb(255, 0, 0);background-color:rgb(255, 255, 255);border: 2px solid rgb(255, 0, 0);border-radius: 5px}")




if __name__ == '__main__':
    app = QApplication(sys.argv)
    path     = str( Path(__file__).absolute() )

    operating_system = platform.system()
    if operating_system in ['windows', 'Windows']:
        temp      = path.split('\\')
        temp[-1]  = 'window_icon.png'
        temp_icon = '\\'.join(temp)
    elif operating_system in ['mac', 'Mac', 'Darwin', 'darwin']:
        temp      = path.split('/')
        temp[-1]  = 'window_icon.icns'
        temp_icon = '/'.join(temp)
    else:
        print('operating system not any of the possible options')
        print('current operating system is: ', operating_system)
        sys.exit()
        
    app.setWindowIcon(QIcon(temp_icon))
    win = DataProcessorGUI()
    win.show()
    app.exec()