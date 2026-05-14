from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import Qt
from PyQt5 import uic
from pathlib import Path


class BrowseInfoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        ui_path = Path(__file__).parent / 'browse_info_dialog.ui'
        uic.loadUi(str(ui_path), self)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        self.infoLabel.setText(
            '<b><span style="color: orange;">IMPORTANT</span></b><br><br>'
            'Make sure that both x and y data you are importing are calibrated and are both in meters.'
        )

        self.acknowledgeButton.clicked.connect(self.accept)
