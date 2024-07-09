from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QPixmap
import cv2
import sys
import time
import numpy as np


class ImageEditorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GIPFOCUS")
        self.setGeometry(100, 100, 800, 800)

        # Cargar y redimensionar el logo
        logo_path = "./GUI/icons/logo.png"
        logo_pixmap = QPixmap(logo_path).scaled(
            64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        if logo_pixmap.isNull():
            print(f"Error: No se pudo cargar el logo desde la ruta {logo_path}")
        else:
            self.setWindowIcon(QIcon(logo_pixmap))

        self.drawing = False
        self.pencil_mode = False
        self.last_point = QPoint()
        self.brush_size = 20
        self.brush_color = Qt.black
        self.currentTool = None
        self.counter = 0
        self.initUI()

    def initUI(self):

        self.setMouseTracking(True)
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
        pencil_button.triggered.connect(self.use_pencil)
        toolbar_top.addAction(pencil_button)

        # Área central para la imagen
        self.label_image = QLabel(self)
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setStyleSheet("background-color: black;")
        self.label_image.setFixedSize(700, 600)
        canvas = QPixmap(700, 600)  # Adjust size as needed
        canvas.fill(Qt.white)  # Fill the canvas with white background
        self.label_image.setPixmap(canvas)
        print("Initial image: ", self.label_image.pixmap())

        # Sección derecha
        self.create_side_section()

        # Layout principal
        self.main_layout = QHBoxLayout()
        self.image_layout = QVBoxLayout()
        self.image_layout.addWidget(toolbar_top)
        self.image_layout.addWidget(self.label_image)

        self.main_layout.addLayout(self.image_layout)
        self.main_layout.addWidget(self.side_section)

        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

    def create_side_section(self):
        # Sección derecha principal
        self.side_section = QWidget()
        side_layout = QVBoxLayout()

        # Primera sección: botones de opciones
        option_group = QGroupBox("FILTROS")
        option_layout = QVBoxLayout()

        options = ["Seg. Semántica", "Filtro 1", "Filtro 2", "Filtro 3", "Filtro 4"]
        self.checkboxes = []

        for option in options:
            checkbox = QCheckBox(option)
            self.checkboxes.append(checkbox)
            option_layout.addWidget(checkbox)

        option_group.setLayout(option_layout)
        side_layout.addWidget(option_group)

        # Botones "Aplicar" y "Visualizar"
        apply_button = QPushButton("Aplicar")
        apply_button.clicked.connect(
            self.apply_filters
        )  # Método que define la acción del botón
        visualize_button = QPushButton("Visualizar")
        visualize_button.clicked.connect(
            self.visualize_filters
        )  # Método que define la acción del botón

        side_layout.addWidget(apply_button)
        side_layout.addWidget(visualize_button)

        # Segunda sección: vista previa de edición
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: black;")
        self.preview_label.setFixedSize(600, 300)

        side_layout.addWidget(self.preview_label)
        self.side_section.setLayout(side_layout)

    # Métodos para manejar las acciones de los botones "Aplicar" y "Visualizar"
    def apply_filters(self):
        # Implementa la lógica para aplicar los filtros seleccionados
        # Implementa la lógica para aplicar los filtros seleccionados
        pass

    def visualize_filters(self):
        # Implementa la lógica para visualizar los filtros seleccionados
        # Implementa la lógica para visualizar los filtros seleccionados
        pass

    def load_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Archivos de Imagen (*.jpg *.jpeg *.png)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            file_name = file_dialog.selectedFiles()[0]
            #pixmap = QPixmap(file_name)
            #pixmap = QPixmap(700, 600)  # Adjust size as needed
            #pixmap.fill(Qt.blue)
            pixmap = self.convert_cv_qt(file_name, saveSize=True)
            self.label_image.setPixmap(pixmap)
            print("New image to: ", self.label_image.pixmap())
            # self.label_image.setScaledContents(True) # Esto no funciona adecuadamente
            # print("New image to (After scale): ", self.label_image.pixmap())
            self.reset_meths()
            # self.main_layout.update()
            #main_layout

    def save_image(self):
        pixmap = self.label_image.pixmap()
        if pixmap:
            save_dialog = QFileDialog(self)
            save_dialog.setAcceptMode(QFileDialog.AcceptSave)
            save_dialog.setNameFilter("Archivos de Imagen (*.jpg *.jpeg *.png)")
            save_dialog.setDefaultSuffix("png")

            if save_dialog.exec_():
                save_path = save_dialog.selectedFiles()[0]
                # Opcion 1 - Guardar con el tamaño original
                a = self.convert_qt_cv(pixmap, setResize=True)
                pixmap = self.convert_cv_qt(a, setResize=False)
                
                # Opcion 2 - Guardar con el tamaño actual de (700, 600)
                # pixmap = self.label_image.pixmap()
                
                if pixmap.save(save_path):
                    QMessageBox.information(
                        self, "Guardar Imagen", "Imagen guardada exitosamente."
                    )
                else:
                    QMessageBox.warning(
                        self, "Guardar Imagen", "Error al guardar la imagen."
                    )
        else:
            QMessageBox.warning(self, "Guardar Imagen", "No hay imagen para guardar.")

    def reset_meths(self):
        self.drawing = False
        self.last_point = QPoint()
        self.brush_size = 20
        self.brush_color = Qt.black
        self.currentTool = None
        self.counter = 0

    def select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.brush_color = color

    def select_brush_size(self):
        size, ok = QInputDialog.getInt(self, "Seleccionar Tamaño del Pincel", "Tamaño del Pincel:", self.brush_size, 1, 50, 1)
        if ok:
            self.brush_size = size

    def use_pencil(self):
        self.currentTool = "pencil"
        print("Current tool: Pencil")
        
        """print("Counter: ", self.counter)
        if self.counter ==1:
            pixmap = QPixmap(700, 600)  # Adjust size as needed
            pixmap.fill(Qt.yellow)
            self.label_image.setPixmap(pixmap)
        elif self.counter == 2:
            pixmap = QPixmap(700, 600)
            pixmap.fill(Qt.green)
            self.label_image.setPixmap(pixmap)
        elif self.counter == 3:
            pixmap = QPixmap(700, 600)
            pixmap.fill(Qt.white)
            self.label_image.setPixmap(pixmap)
        self.counter += 1"""

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.currentTool == "pencil":
            print("Mouse pressed")
            self.drawing = True
            # Convert global position to local position relative to label_image
            localPos = self.label_image.mapFromGlobal(event.globalPos())
            self.lastPoint = localPos

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing and self.currentTool == "pencil":
            print("Mouse is moving...")
            pixmap = self.label_image.pixmap()
            print(self.label_image.pixmap())
            painter = QPainter(pixmap)
            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            # Convert global position to local position relative to label_image
            localPos = self.label_image.mapFromGlobal(event.globalPos())
            painter.drawLine(self.lastPoint, localPos)
            self.lastPoint = localPos
            self.label_image.setPixmap(pixmap)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing and self.currentTool == "pencil":
            print("Mouse released")
            self.drawing = False
            # self.label_image.setPixmap(self.label_image.pixmap())
            # self.counter += 1
            print("Counter: ", self.counter)
            if self.counter > 3:
                self.save_image()

    # To convert from opencv to QPixmap
    def convert_cv_qt(self, cv_img, saveSize=False, setResize=True):
        """Convert from an opencv image to QPixmap"""
        if isinstance(cv_img, np.ndarray):
            # cv_img is a numpy array
            self.rgb_image = cv_img.astype(np.uint8)
        else:
            image = cv2.imread(cv_img)
            self.rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if saveSize:    # Save the original size of the image
            self.original_image_shape = self.rgb_image.shape
        
        if setResize:   # Resize the image to a fixed size
            self.rgb_image = cv2.resize(self.rgb_image, (700, 600))
        
        h, w, ch = self.rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(self.rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # p = convert_to_Qt_format.scaled(700, 600, Qt.KeepAspectRatio) # En caso de querer escalar la imagen QT
        p = convert_to_Qt_format
        return QPixmap.fromImage(p)
    
    def convert_qt_cv(self, qt_img, setResize=False): # To convert from QPixmap to opencv
        """Convert from a QPixmap to an opencv image"""
        qimage = qt_img.toImage()
        if qimage.format() != QImage.Format_RGB888:
            qimage = qimage.convertToFormat(QImage.Format_RGB888)
        b = qimage.bits()
        b.setsize(qimage.byteCount())
        stride = qimage.bytesPerLine()
        # cv_img = np.array(b).reshape(qimage.height(), qimage.width(), 3)
        cv_img = np.frombuffer(b, dtype=np.uint8).reshape((qimage.height(), stride//3, 3))
        if setResize:
            cv_img = cv2.resize(cv_img, (self.original_image_shape[1], self.original_image_shape[0]))
        return cv_img


class SplashScreen(QSplashScreen):
    def __init__(self, pixmap):
        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet(
            "QProgressBar {border: 1px solid black;text-align: top;padding: 2px;border-radius: 7px;"
            "background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1,stop: 0 #fff,stop: 0.4999 #eee,stop: 0.5 #ddd,stop: 1 #eee );height: 11px}"
            "QProgressBar::chunk {background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0,stop: 0 #ea00ff,stop: 1 #68fdff );"
            "border-top-left-radius: 7px;border-radius: 7px;border: None;}"
        )
        self.progress_bar.setFixedWidth(530)
        self.progress_bar.move(120, 570)
        self.Loading_text = QLabel(self)
        self.Loading_text.setFont(QFont("Calibri", 11))
        self.Loading_text.setStyleSheet(
            "QLabel { background-color : None; color : #c12cff; }"
        )
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

    texts = [
        "Initializing...",
        "Getting path...",
        "Measuring memory...",
        "Scanning for plugs in...",
        "Initializing panels...",
        "Loading library...",
        "Building color conversion tables...",
        "Reading tools...",
        "Reading Preferences...",
        "Getting ready...",
    ]
    for i in range(0, 101):
        splash.update_progress(i, texts[min(i // 10, len(texts) - 1)])
    time.sleep(1)
