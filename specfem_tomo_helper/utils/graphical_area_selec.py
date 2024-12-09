import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from specfem_tomo_helper.utils.pyqt_gui import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
