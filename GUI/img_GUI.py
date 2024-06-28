from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import time

class ImageEditorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GIPFOCUS")
        self.setGeometry(100, 100, 800, 600)

        # Cargar y redimensionar el logo
        logo_path = ".\GUI\icons\logo.png"  # Asegúrate de que esta ruta sea correcta
        logo_pixmap = QPixmap(logo_path).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Verificar que el logo se carga correctamente
        if logo_pixmap.isNull():
            print(f"Error: No se pudo cargar el logo desde la ruta {logo_path}")
        else:
            self.setWindowIcon(QIcon(logo_pixmap))

        self.initUI()

    def initUI(self):
        # Menú
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Archivo")

        insert_action = QAction("Insertar", self)
        insert_action.triggered.connect(self.load_image)
        file_menu.addAction(insert_action)

        save_action = QAction("Guardar", self)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)

        # Barra de herramientas superior
        toolbar_top = QToolBar("Herramientas Superior")
        self.addToolBar(Qt.TopToolBarArea, toolbar_top)

        color_button = QAction(QIcon(), "Paleta de Colores", self)
        color_button.triggered.connect(self.select_color)
        toolbar_top.addAction(color_button)

        pencil_button = QAction(QIcon(), "Lápiz", self)
        toolbar_top.addAction(pencil_button)

        paint_button = QAction(QIcon(), "Pintura", self)
        toolbar_top.addAction(paint_button)

        crop_button = QAction(QIcon(), "Recortar", self)
        toolbar_top.addAction(crop_button)

        # Área central para la imagen
        self.label_image = QLabel(self)
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setStyleSheet("background-color: black;")
        self.label_image.setFixedSize(500, 400)

        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.addWidget(toolbar_top)  # Agregar la barra de herramientas superior
        main_layout.addWidget(self.label_image)  # Agregar el área de la imagen

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def load_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Archivos de Imagen (*.jpg *.jpeg *.png)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            file_name = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(file_name)
            self.label_image.setPixmap(pixmap)
            self.label_image.setScaledContents(True)

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

    def select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            print(f"Color seleccionado: {color.name()}")

class SplashScreen(QSplashScreen):
    def __init__(self, pixmap):
        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("QProgressBar {border: 1px solid black;text-align: top;padding: 2px;border-radius: 7px;"
                                        "background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1,stop: 0 #fff,stop: 0.4999 #eee,stop: 0.5 #ddd,stop: 1 #eee );height: 11px}"
                                        "QProgressBar::chunk {background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0,stop: 0 #ea00ff,stop: 1 #68fdff );"
                                        "border-top-left-radius: 7px;border-radius: 7px;border: None;}")
        self.progress_bar.setFixedWidth(530)
        self.progress_bar.move(120, 570)
        self.Loading_text = QLabel(self)
        self.Loading_text.setFont(QFont("Calibri", 11))
        self.Loading_text.setStyleSheet("QLabel { background-color : None; color : #c12cff; }")
        self.Loading_text.setGeometry(125, 585, 300, 50)

    def update_progress(self, i, text):
        self.Loading_text.setText(text)
        self.progress_bar.setValue(i)
        t = time.time()
        while time.time() < t + 0.035:
            QApplication.processEvents()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    splash_pix = QPixmap("TP_assets/splashscreen3.png")
    splash = SplashScreen(splash_pix)
    splash.show()

    texts = ["Initializing...", "Getting path...", "Measuring memory...", "Scanning for plugs in...", "Initializing panels...", 
             "Loading library...", "Building color conversion tables...", "Reading tools...", "Reading Preferences...", "Getting ready..."]
    for i in range(0, 101):
        splash.update_progress(i, texts[min(i // 10, len(texts) - 1)])
    time.sleep(1)
