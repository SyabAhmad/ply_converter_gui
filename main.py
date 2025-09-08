import sys
from PyQt5.QtWidgets import QApplication
from gui import PLYConverterGUI

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("PLY Converter")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("PLY Converter")
    window = PLYConverterGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()