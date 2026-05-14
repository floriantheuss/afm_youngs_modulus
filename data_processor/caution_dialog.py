from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import Qt
from PyQt5 import uic
from pathlib import Path


class CautionDialog(QDialog):
    def __init__(self, caution_list, parent=None):
        super().__init__(parent)
        ui_path = Path(__file__).parent / 'caution_dialog.ui'
        uic.loadUi(str(ui_path), self)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        num = len(caution_list)
        self.cautionLabel.setText(
            f'<b><span style="color: red;">CAUTION!</span></b><br><br>'
            f'<span style="color: red;">{num}</span> instance(s) where the z-sensor increases slower than the deflection have been noticed.<br><br>'
            'This could be an indicator that the deflection is not in m units. It may be in volts and needs to be multiplied by a calibration factor. Both deflection and z-sensor need to be in the same length unit.'
        )
        for item in caution_list:
            self.cautionListWidget.addItem(f'x index: {item[0]},  y index: {item[1]}')

        self.acknowledgeButton.clicked.connect(self.accept)


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    dialog = CautionDialog([(0, 1), (2, 3), (4, 5), (2, 3), (4, 5), (2, 3), (4, 5), (2, 3), (4, 5)])
    dialog.exec_()
    sys.exit()
