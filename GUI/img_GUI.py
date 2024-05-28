from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QMessageBox, QPushButton, QFileDialog
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

        btn_save_image = QPushButton("Guardar Imagen", self)
        btn_save_image.setGeometry(160, 10, 100, 30)
        btn_save_image.clicked.connect(self.save_image)

    def load_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Archivos de Imagen (*.jpg *.jpeg *.png)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            file_name = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(file_name)
            self.label_image.setPixmap(pixmap)

    def save_image(self):
        pixmap = self.label_image.pixmap()
        if pixmap:
            save_dialog = QFileDialog(self)
            save_dialog.setAcceptMode(QFileDialog.AcceptSave)
            save_dialog.setNameFilter("Archivos de Imagen (*.jpg *.jpeg *.png)")
            save_dialog.setDefaultSuffix('png')

            if save_dialog.exec_():
                save_path = save_dialog.selectedFiles()[0]
                if pixmap.save(save_path):
                    QMessageBox.information(self, "Guardar Imagen", "Imagen guardada exitosamente.")
                else:
                    QMessageBox.warning(self, "Guardar Imagen", "Error al guardar la imagen.")
        else:
            QMessageBox.warning(self, "Guardar Imagen", "No hay imagen para guardar.")
    
