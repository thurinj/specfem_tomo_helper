import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QLineEdit, QPushButton, QRadioButton, QButtonGroup, QToolTip
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Geographical Area Selection")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.create_widgets()
        self.connect_signals()

    def create_widgets(self):
        self.latitude_label = QLabel("Latitude:")
        self.latitude_slider = QSlider(Qt.Horizontal)
        self.latitude_slider.setMinimum(-90)
        self.latitude_slider.setMaximum(90)
        self.latitude_slider.setValue(0)
        self.latitude_slider.setToolTip("Adjust the latitude")

        self.longitude_label = QLabel("Longitude:")
        self.longitude_slider = QSlider(Qt.Horizontal)
        self.longitude_slider.setMinimum(-180)
        self.longitude_slider.setMaximum(180)
        self.longitude_slider.setValue(0)
        self.longitude_slider.setToolTip("Adjust the longitude")

        self.easting_label = QLabel("Easting:")
        self.easting_input = QLineEdit()
        self.easting_input.setToolTip("Enter the easting value")

        self.northing_label = QLabel("Northing:")
        self.northing_input = QLineEdit()
        self.northing_input.setToolTip("Enter the northing value")

        self.utm_zone_label = QLabel("UTM Zone:")
        self.utm_zone_input = QLineEdit()
        self.utm_zone_input.setToolTip("Enter the UTM zone")

        self.hemisphere_label = QLabel("Hemisphere:")
        self.north_radio = QRadioButton("North")
        self.south_radio = QRadioButton("South")
        self.hemisphere_group = QButtonGroup()
        self.hemisphere_group.addButton(self.north_radio)
        self.hemisphere_group.addButton(self.south_radio)
        self.north_radio.setChecked(True)
        self.north_radio.setToolTip("Select the northern hemisphere")
        self.south_radio.setToolTip("Select the southern hemisphere")

        self.export_button = QPushButton("Export Values")
        self.export_button.setToolTip("Export the selected values")

        self.layout.addWidget(self.latitude_label)
        self.layout.addWidget(self.latitude_slider)
        self.layout.addWidget(self.longitude_label)
        self.layout.addWidget(self.longitude_slider)
        self.layout.addWidget(self.easting_label)
        self.layout.addWidget(self.easting_input)
        self.layout.addWidget(self.northing_label)
        self.layout.addWidget(self.northing_input)
        self.layout.addWidget(self.utm_zone_label)
        self.layout.addWidget(self.utm_zone_input)
        self.layout.addWidget(self.hemisphere_label)
        self.layout.addWidget(self.north_radio)
        self.layout.addWidget(self.south_radio)
        self.layout.addWidget(self.export_button)

    def connect_signals(self):
        self.latitude_slider.valueChanged.connect(self.update_latitude)
        self.longitude_slider.valueChanged.connect(self.update_longitude)
        self.easting_input.textChanged.connect(self.update_easting)
        self.northing_input.textChanged.connect(self.update_northing)
        self.utm_zone_input.textChanged.connect(self.update_utm_zone)
        self.hemisphere_group.buttonClicked.connect(self.update_hemisphere)
        self.export_button.clicked.connect(self.export_values)

    def update_latitude(self, value):
        print(f"Latitude: {value}")

    def update_longitude(self, value):
        print(f"Longitude: {value}")

    def update_easting(self, value):
        print(f"Easting: {value}")

    def update_northing(self, value):
        print(f"Northing: {value}")

    def update_utm_zone(self, value):
        print(f"UTM Zone: {value}")

    def update_hemisphere(self, button):
        print(f"Hemisphere: {button.text()}")

    def export_values(self):
        latitude = self.latitude_slider.value()
        longitude = self.longitude_slider.value()
        easting = self.easting_input.text()
        northing = self.northing_input.text()
        utm_zone = self.utm_zone_input.text()
        hemisphere = self.hemisphere_group.checkedButton().text()

        print(f"Latitude: {latitude}")
        print(f"Longitude: {longitude}")
        print(f"Easting: {easting}")
        print(f"Northing: {northing}")
        print(f"UTM Zone: {utm_zone}")
        print(f"Hemisphere: {hemisphere}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
