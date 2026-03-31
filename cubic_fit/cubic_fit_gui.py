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
from cubic_fit.cubic_fit import CubicFit


class CubicFitGUI (QMainWindow):
    def __init__(self,operating_system=None):
        super(CubicFitGUI, self).__init__()
        
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
        temp[-1] = 'cubic_fit_gui.ui'
        temp_ui     = self.separator.join(temp)
        uic.loadUi(temp_ui, self)
        
        temp[-1]  = icon_name
        temp_icon = self.separator.join(temp)
        # self.setWindowIcon(QIcon(temp_icon))

        self.initialize_data_and_fit_variables()
        self.initialize_directory_widgets()
        self.create_compliance_map_plot()
        self.initialize_compliance_map_widgets()
        self.create_individual_data_plot()
        self.initialize_cubic_fit_widgets()
        self.create_fit_plot()
           
    def initialize_directory_widgets(self):
        # Widgets and Variables for loading files
        self.fileDialog = QFileDialog()
        self.loadComplianceMapButton.clicked.connect(self.load_compliance_map)
        self.browseRawDataButton.clicked.connect(self.browse_raw_data)
        self.complianceMapPath = None
        self.rawDataDir = None
        self.dir = ''


    def initialize_data_and_fit_variables(self):
        # variables to store data and fit_parameters
        self.afmData             = AFMForceMapData()
        self.retract_data        = None
        self.approach_data       = None
        self.fit_data            = None
        self.fit                 = None
        self.compliance_map      = None
        self.radius              = 5
        self.circle_center       = np.array([5,5])
        self.radius_um           = 1
        self.scan_win_size       = None
        self.k_tip               = None
        self.Elin                = None
        self.Ecub                = None
        self.x_shift             = None
        self.y_shift             = None
        self.thickness           = None
        self.tension             = None
        self.poisson             = None
    
    def load_compliance_map (self):    
        try:
            options = QFileDialog.Options()
            self.complianceMapPath, _ = self.fileDialog.getOpenFileName(self, "Open Compliance Map",self.dir,"All Files (*)", options=options)
            self.dir = self.separator.join(self.complianceMapPath.split(self.separator)[:-1])
            
            data = np.load(self.complianceMapPath)
            self.compliance_map = data['compliance map (m/N)']
            self.radius         = data['radius (pixel)']
            self.circle_center  = data['circle center (pixels)']
            self.radius_um      = data['radius (um)'] * 1e-6
            self.scan_win_size  = data['scan window size (um)'] * 1e-6
            self.k_tip          = data['tip spring constant']
            
            self.complianceMapLine.setText(self.complianceMapPath)
            self.circleCenterlineEdit.setText(f'{np.round(self.circle_center[0],0)}, {np.round(self.circle_center[1],0)}')
            self.currentPositionEdit.setText(f'{np.round(self.circle_center[0],0)}, {np.round(self.circle_center[1],0)}')
            self.radiusLine.setText(f'{np.round(self.radius_um*1e6,3)}')
            self.kTipLine.setText(f'{np.round(self.k_tip,3)}')
            
            self.update_compliance_map_plot(compliance_map=self.compliance_map)
            self.circle_center_plot.setPos(int(self.circle_center[0]), int(self.circle_center[1]))
            self.update_circle_params_center()
        except Exception as e:
            print('Error loading compliance map ...')
            print(e)

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
      
        self.circle = QGraphicsEllipseItem(self.circle_center[0] - self.radius, self.circle_center[1] - self.radius, 2*self.radius, 2*self.radius)
        self.circle.setPen(pg.mkPen(color='r', width=2))  # Set the pen color and width
        self.complianceMapPlot.addItem(self.circle)

    def update_compliance_map_plot (self, compliance_map):
        self.compliancemap_imageItem.setImage(compliance_map.T)
        # self.complianceMapPlot.setRange(rect=self.compliancemap_imageItem.boundingRect())
        self.complianceMapPlot.setRange(xRange=(0,np.shape(compliance_map)[0]),yRange=(0,np.shape(compliance_map)[1]), padding=0)
        self.active_compliance_map = compliance_map

    def browse_raw_data (self):    
        try:
            if self.complianceMapPath is None:
                directory = ''
            else:
                directory = self.separator.join(self.complianceMapPath.split(self.separator)[:-1])
            
            self.rawDataDir = self.fileDialog.getExistingDirectory(self, "Select Raw Data Folder", directory)
            x_name = self.xnamelineEdit.text().strip()
            y_name = self.ynamelineEdit.text().strip()
            self.afmData.get_filenames(self.rawDataDir, x_name, y_name)
            self.afmData.load_data()
            self.rawDataLine.setText(self.rawDataDir)
            
        except Exception as e:
            print('Error loading raw data directory ...')
            print(e)
    
    def initialize_compliance_map_widgets (self):    
        self.circle_center_plot.sigPositionChanged.connect(self.update_circle_params_center)
    
    def create_individual_data_plot (self):
        legend_item = self.rawDataPlot.addLegend(frame=False, labelTextColor='w', labelTextSize='14pt')

        self.approach_data_plot = pg.PlotCurveItem([],[])
        self.approach_data_plot.setPen(pg.mkPen(color='g', width=3))#, style=pg.QtCore.Qt.DashLine))
        self.rawDataPlot.addItem(self.approach_data_plot)
        legend_item.addItem(self.approach_data_plot, name='approach')

        self.retract_data_plot = pg.PlotCurveItem([],[])
        self.retract_data_plot.setPen(pg.mkPen(color='red', width=3))#, style=pg.QtCore.Qt.DashLine))
        self.rawDataPlot.addItem(self.retract_data_plot)
        legend_item.addItem(self.retract_data_plot, name='retract')

        self.boundLine1 = pg.InfiniteLine(pos=1, angle=90, movable=True, pen='w')
        self.rawDataPlot.addItem(self.boundLine1)
        self.boundLine2 = pg.InfiniteLine(pos=2, angle=90, movable=True, pen='w')
        self.rawDataPlot.addItem(self.boundLine2)

        # self.rawDataPlot.setTitle('Cubic Fit', **{'color': '#FFF', 'size': '18pt'})    

        self.rawDataPlot.showAxis('top', show=True)
        self.rawDataPlot.showAxis('right', show=True)
        self.rawDataPlot.getAxis('top').setStyle(showValues=False)
        self.rawDataPlot.getAxis('right').setStyle(showValues=False)
        self.rawDataPlot.setLabel('left', 'AFM Tip Deflection', units='m', **{'color': '#FFF', 'font-size': '12pt'})
        self.rawDataPlot.setLabel('bottom', 'Z-Sensor Distance', units='m', **{'color': '#FFF', 'font-size': '12pt'})

    def initialize_cubic_fit_widgets (self):
        self.plotSelectedDataButton.clicked.connect(self.plot_selected_data_button_clicked)
        self.boundLine1.sigPositionChanged.connect(self.update_fit_plot_data)
        self.boundLine2.sigPositionChanged.connect(self.update_fit_plot_data)
        self.fitDataBox.currentTextChanged.connect(self.update_fit_plot_data)

        self.fitCubicButton.clicked.connect(self.perform_cubic_fit)
    
    def update_circle_params_center (self):
        pos = np.array([self.circle_center_plot.pos().x(),self.circle_center_plot.pos().y()])
        diff = -self.circle_center+pos
        self.circle_center = pos
        self.currentPositionEdit.setText(f'{self.circle_center[0]:.2f}, {self.circle_center[1]:.2f}')

        self.circle.setRect(self.circle_center[0] - self.radius, self.circle_center[1] - self.radius, 2*self.radius, 2*self.radius)

    def plot_selected_data_button_clicked (self):
        x = int(np.round(self.circle_center_plot.pos().x(), 0))
        y = int(np.round(self.circle_center_plot.pos().y(), 0))
        # making the senter of the circle integer pixels
        # not sure if this is super correct, but we definitely don't have more resolution than 1 pixel
        # but not sure if just rounding to the next integer is the correct approach
        # but then again shouldn't matter much
        self.circle_center_plot.setPos(x, y)
        self.update_circle_params_center()

        indices = self.afmData.x_index
        boolean_array = (indices==np.array([y,x]))
        oneD_mask = boolean_array.all(axis=1)
        index = np.arange(len(indices))[oneD_mask]
        index = index[0]
        # update plot
        self.approach_data = self.afmData.approach_data[index]
        self.retract_data = self.afmData.retract_data[index]
        self.approach_data_plot.setData(self.approach_data[0], self.approach_data[1])
        self.retract_data_plot.setData(self.retract_data[0], self.retract_data[1])

        # add vertical lines - use as bounds for fit
        pos1 = np.min(self.approach_data[0]) + 1/3 * (np.max(self.approach_data[0]) - np.min(self.approach_data[0]))
        self.boundLine1.setValue(pos1)
        pos2 = np.min(self.approach_data[0]) + 2/3 * (np.max(self.approach_data[0]) - np.min(self.approach_data[0]))
        self.boundLine2.setValue(pos2)

        self.fit_plot.setData([],[])

    def create_fit_plot (self):
        legend_item = self.cubicFitPlot.addLegend(frame=False, labelTextColor='w', labelTextSize='14pt')

        self.fit_data_plot = pg.PlotCurveItem([],[])
        self.fit_data_plot.setPen(pg.mkPen(color='w', width=3))
        self.cubicFitPlot.addItem(self.fit_data_plot)
        legend_item.addItem(self.fit_data_plot, name='data')

        self.fit_plot = pg.PlotCurveItem([],[])
        self.fit_plot.setPen(pg.mkPen(color='red', width=2, style=pg.QtCore.Qt.DashLine))
        self.cubicFitPlot.addItem(self.fit_plot)
        legend_item.addItem(self.fit_plot, name='fit')

        self.cubicFitPlot.showAxis('top', show=True)
        self.cubicFitPlot.showAxis('right', show=True)
        self.cubicFitPlot.getAxis('top').setStyle(showValues=False)
        self.cubicFitPlot.getAxis('right').setStyle(showValues=False)
        self.cubicFitPlot.setLabel('left', 'Force', units='N', **{'color': '#FFF', 'font-size': '12pt'})
        self.cubicFitPlot.setLabel('bottom', 'Deflection', units='m', **{'color': '#FFF', 'font-size': '12pt'})

    def update_fit_plot_data (self):
        try:
            text = self.fitDataBox.currentText()
            if text == 'approach':
                x, y = self.approach_data
            else:
                x, y = self.retract_data

            bounds = np.array([self.boundLine1.value(), self.boundLine2.value()])
            mask   = (x>=np.min(bounds)) & (x<=np.max(bounds))
            x, y   = x[mask], y[mask]
            x      = x - np.min(x)
            y      = y - np.min(y)

            # convert x, y to correct units
            y = self.k_tip*y
            self.fit_data = np.array([x, y])

            self.fit_data_plot.setData(x, y)
            self.fit_plot.setData([],[])
        except Exception as e:
            print('Error updating fit plot data ...')
            print(e)

    def perform_cubic_fit (self):
        try:
            self.clear_results()
            
            self.thickness = float(self.thicknessLine.text()) *1e-9
            self.poisson   = float(self.poissonLine.text())
            self.tension   = float(self.tensionLine.text())
            linear_guess = float(self.ELinGuessEdit.text()) * 1e9
            cubic_guess  = float(self.ECubGuessEdit.text()) * 1e9
            x_guess      = float(self.xShiftGuessEdit.text())
            y_guess      = float(self.yShiftGuessEdit.text())
            guess = np.array([linear_guess, cubic_guess, x_guess, y_guess])
            vary  = np.array([self.ELinBox.isChecked(), self.ECubBox.isChecked(), self.xShiftBox.isChecked(), self.yShiftBox.isChecked()])
            
            self.fit = CubicFit(self.fit_data, self.thickness, self.radius_um, self.tension, self.poisson)
            popt, fit_plot = self.fit.perform_fit(guess=guess, vary=vary)
            self.Elin    = popt[0]
            self.Ecub    = popt[1]
            self.x_shift = popt[2]
            self.y_shift = popt[3]
            
            self.ELinLineEdit.setText(f'{np.round(self.Elin/1e9, 3)}')
            self.ECubLineEdit.setText(f'{np.round(self.Ecub/1e9, 3)}')
            self.xShiftLine.setText(f'{self.x_shift:.3e}')
            self.yShiftLine.setText(f'{self.y_shift:.3e}')
            self.fit_plot.setData(fit_plot[0],fit_plot[1])

        except Exception as e:
            print('Error fitting data ...')
            print(e)

    def clear_results (self):
        self.ELinLineEdit.setText('')
        self.ECubLineEdit.setText('')
        self.xShiftLine.setText('')
        self.yShiftLine.setText('')
        self.Elin = None
        self.Ecub = None
        self.x_shift = None
        self.y_shift = None