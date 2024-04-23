# Author: Daniela Cabrales
# Otto
import sys
from img_GUI import ImageEditorApp
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    window = ImageEditorApp()
    window.show()
    sys.exit(app.exec_())
    return 0

if __name__ == "__main__":
    main()
