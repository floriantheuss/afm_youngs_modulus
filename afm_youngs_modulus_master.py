from PyQt5.QtWidgets import (QApplication,
                             QMainWindow,
                             QPushButton,
                             QHBoxLayout,
                             QVBoxLayout,
                             QWidget,
                             QFileDialog,
                             QTableWidget,
                             QTableWidgetItem,
                             QDockWidget)
from PyQt5.QtCore import QTimer
from PyQt5 import uic, QtCore
from PyQt5.QtGui import QIcon
from pathlib import Path
from data_processor.afm_data_processor_gui import DataProcessorGUI
from compliance_fit.compliance_fit_gui import ComplianceFitGUI
from cubic_fit.cubic_fit_gui import CubicFitGUI
from copy import deepcopy
import platform
import sys



class AFMYoungsModulusMaster (QMainWindow):
    def __init__(self):
        super(AFMYoungsModulusMaster, self).__init__()

        # different folders in file paths are separated by different symbols depending on the operating software
        # which one it is is important to know when creating correct paths to load/save data
        self.operating_system = platform.system()
        if self.operating_system in ['windows', 'Windows']:
            separator = '\\'
        elif self.operating_system in ['mac', 'Mac', 'Darwin', 'darwin']:
            separator = '/'
        else:
            print('operating system not any of the possible options')
            print('current operating system is: ', self.operating_system)


        # import ui file
        path     = str( Path(__file__).absolute() )
        temp     = path.split(separator)
        temp[-1] = 'afm_youngs_modulus_master.ui'
        temp_ui     = separator.join(temp)
        uic.loadUi(temp_ui, self)

        # initialize data processor
        # I am initializing all these classes here (could have just done it once the button is clicked)
        # because this way, if the specific window is closed, the class still exists
        self.dataProcessorWindow = DataProcessorGUI(operating_system=self.operating_system)
        self.DataProcessorButton.clicked.connect(self.open_data_processor)

        # here I am not immediately initializing the class, because that entails creating a mesh which takes some time
        # I am still achieving what we have above with the modified "open_compliance_fit_module" fct
        self.complianceFitWindow = None
        self.ComplianceFitModuleButton.clicked.connect(self.open_compliance_fit_module)

        # here I am not immediately initializing the class, it is starting to be too much and the cubic fit is also really something that shouldn't be done much
        # I am still achieving what we have above with the modified "open_compliance_fit_module" fct
        self.cubicFitWindow = None
        self.CubicFitButton.clicked.connect(self.open_cubic_fit_window)

        # list of all windows so we can close them all when closing this window
        self.windows = [self.dataProcessorWindow]

        # Connect the close event to the custom function
        self.closeEvent = self.on_close_event

    def open_data_processor(self):
        if self.dataProcessorWindow is None: # asking this ensures that we can open and close this window without losing anything
            self.dataProcessorWindow = DataProcessorGUI(operating_system=self.operating_system)
        self.dataProcessorWindow.show()
    
    def open_compliance_fit_module(self):
        # if self.complianceFitWindow is None:
        # here we will just initialize a completely new instance of the class;
        # for some reason the pyvista plotter had problems otherwise
        # self.complianceFitWindow = ComplianceFitGUI(operating_system=self.operating_system)
        # self.complianceFitWindow.show()
        if self.complianceFitWindow is None: # asking this ensures that we can open and close this window without losing anything
            self.complianceFitWindow = ComplianceFitGUI(operating_system=self.operating_system)
            self.windows.append(self.complianceFitWindow)
        else:
            # this is a pyvista plot which doesn't like it when the window is closed
            # so what I do is close the pyvista plot every time the window is closed
            # but then if have to reinitialize it here
            self.complianceFitWindow.create_compliance_map_plot()
            self.complianceFitWindow.update_compliance_map_plot()
        self.complianceFitWindow.show()
    
    def open_cubic_fit_window(self):
        if self.cubicFitWindow is None: # asking this ensures that we can open and close this window without losing anything
            self.cubicFitWindow = CubicFitGUI(operating_system=self.operating_system)
            self.windows.append(self.cubicFitWindow)
        self.cubicFitWindow.show()

    def on_close_event(self, event):
        # if this window is closed, also close all other windows opened by this one
        for window in self.windows:
            if not window is None:
                window.close()

        # Call the default closeEvent to close the window
        super().closeEvent(event)



if __name__ == '__main__':
    # QApplication.setStyle('Fusion')
    app = QApplication([])
    win = AFMYoungsModulusMaster()
    win.show()
    app.exec()