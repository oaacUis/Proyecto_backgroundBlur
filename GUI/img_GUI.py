from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap

class ImageEditorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 600, 400)

        self.initUI()

    def initUI(self):
        self.label_image = QLabel(self)
        self.label_image.setGeometry(50, 50, 500, 300)
        self.label_image.setScaledContents(True)

        btn_load_image = QPushButton("Cargar Imagen", self)
        btn_load_image.setGeometry(50, 10, 100, 30)
        btn_load_image.clicked.connect(self.load_image)

    def load_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Archivos de Imagen (*.jpg *.jpeg *.png)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            file_name = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(file_name)
            self.label_image.setPixmap(pixmap)
