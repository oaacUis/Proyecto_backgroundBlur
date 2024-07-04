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
        logo_path = "./GUI/icons/logo.png"
        logo_pixmap = QPixmap(logo_path).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        if logo_pixmap.isNull():
            print(f"Error: No se pudo cargar el logo desde la ruta {logo_path}")
        else:
            self.setWindowIcon(QIcon(logo_pixmap))

        self.drawing = False
        self.last_point = QPoint()
        self.brush_size = 3
        self.brush_color = Qt.black
        self.crop_shape = "Rectángulo"  # Forma inicial de recorte
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
        pencil_button.triggered.connect(self.use_pencil)
        toolbar_top.addAction(pencil_button)

        paint_button = QAction(QIcon(), "Pintura", self)
        paint_button.triggered.connect(self.use_paint)
        toolbar_top.addAction(paint_button)

        crop_button = QAction(QIcon(), "Recortar", self)
        crop_button.triggered.connect(self.select_crop_shape)
        toolbar_top.addAction(crop_button)

        # Área central para la imagen
        self.label_image = QLabel(self)
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setStyleSheet("background-color: black;")
        self.label_image.setFixedSize(500, 400)

        # Sección derecha
        self.create_side_section()

        # Layout principal
        main_layout = QHBoxLayout()
        image_layout = QVBoxLayout()
        image_layout.addWidget(toolbar_top)
        image_layout.addWidget(self.label_image)

        main_layout.addLayout(image_layout)
        main_layout.addWidget(self.side_section)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
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
        apply_button.clicked.connect(self.apply_filters)  # Método que define la acción del botón
        visualize_button = QPushButton("Visualizar")
        visualize_button.clicked.connect(self.visualize_filters)  # Método que define la acción del botón

        side_layout.addWidget(apply_button)
        side_layout.addWidget(visualize_button)

        # Segunda sección: vista previa de edición
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: black;")
        self.preview_label.setFixedSize(260, 200)

        side_layout.addWidget(self.preview_label)
        self.side_section.setLayout(side_layout)
    
    # Métodos para manejar las acciones de los botones "Aplicar" y "Visualizar"
    def apply_filters(self):
    # Implementa la lógica para aplicar los filtros seleccionados
        pass

    def visualize_filters(self):
    # Implementa la lógica para visualizar los filtros seleccionados
        pass

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
            self.brush_color = color

    def use_pencil(self):
        self.label_image.mousePressEvent = self.mouse_press
        self.label_image.mouseMoveEvent = self.mouse_move
        self.label_image.mouseReleaseEvent = self.mouse_release

    def use_paint(self):
        pass  # Implementa funcionalidad de pintura

    def select_crop_shape(self):
        items = ["Rectángulo", "Elipse"]  # Opciones de formas de recorte
        item, ok = QInputDialog.getItem(self, "Seleccionar Forma de Recorte", 
                                        "Selecciona la forma de recorte:", items, 0, False)
        if ok and item:
            self.crop_shape = item

    def use_crop(self, event):
        if self.crop_shape == "Rectángulo":
            self.label_image.mousePressEvent = self.rect_crop_mouse_press
            self.label_image.mouseMoveEvent = self.rect_crop_mouse_move
            self.label_image.mouseReleaseEvent = self.rect_crop_mouse_release
        elif self.crop_shape == "Elipse":
            self.label_image.mousePressEvent = self.ellipse_crop_mouse_press
            self.label_image.mouseMoveEvent = self.ellipse_crop_mouse_move
            self.label_image.mouseReleaseEvent = self.ellipse_crop_mouse_release
        
        self.cropping = False
        self.crop_start = QPoint()
        self.crop_rect = QRect()

    def rect_crop_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.cropping = True
            self.crop_start = event.pos()
            self.crop_rect.setTopLeft(self.crop_start)
            self.crop_rect.setBottomRight(self.crop_start)
            self.update_crop_preview()

    def rect_crop_mouse_move(self, event):
        if self.cropping:
            self.crop_rect.setBottomRight(event.pos())
            self.update_crop_preview()

    def rect_crop_mouse_release(self, event):
        if event.button() == Qt.LeftButton and self.cropping:
            self.cropping = False
            self.update_crop_preview()
            self.perform_crop()

    def ellipse_crop_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.cropping = True
            self.crop_start = event.pos()
            self.crop_rect.setTopLeft(self.crop_start)
            self.crop_rect.setBottomRight(self.crop_start)
            self.update_crop_preview()

    def ellipse_crop_mouse_move(self, event):
        if self.cropping:
            self.crop_rect.setBottomRight(event.pos())
            self.update_crop_preview()

    def ellipse_crop_mouse_release(self, event):
        if event.button() == Qt.LeftButton and self.cropping:
            self.cropping = False
            self.update_crop_preview()
            self.perform_crop()

    def update_crop_preview(self):
        pixmap = self.label_image.pixmap()
        if pixmap:
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.red, 1, Qt.SolidLine))
            if self.crop_shape == "Rectángulo":
                painter.drawRect(self.crop_rect)
            elif self.crop_shape == "Elipse":
                painter.drawEllipse(self.crop_rect)
            self.label_image.setPixmap(pixmap)

    def perform_crop(self):
        pixmap = self.label_image.pixmap()
        if pixmap:
            if self.crop_shape == "Rectángulo":
                cropped_pixmap = pixmap.copy(self.crop_rect.normalized())
            elif self.crop_shape == "Elipse":
                mask = QPixmap(pixmap.size())
                mask.fill(Qt.transparent)
                painter = QPainter(mask)
                painter.setBrush(Qt.black)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(self.crop_rect)
                painter.end()

                cropped_pixmap = QPixmap(pixmap.size())
                cropped_pixmap.fill(Qt.transparent)
                painter = QPainter(cropped_pixmap)
                painter.setCompositionMode(QPainter.CompositionMode_Source)
                painter.drawPixmap(0, 0, pixmap)
                painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
                painter.drawPixmap(0, 0, mask)
                painter.end()

            self.label_image.setPixmap(cropped_pixmap)

    def mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()
            self.drawing = True

    def mouse_move(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.label_image.pixmap())
            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.label_image.update()

    def mouse_release(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

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

    texts = ["Initializing...", "Getting path...", "Measuring memory...", "Scanning for plugs in...",
             "Initializing panels...", "Loading library...", "Building color conversion tables...",
             "Reading tools...", "Reading Preferences...", "Getting ready..."]
    for i in range(0, 101):
        splash.update_progress(i, texts[min(i // 10, len(texts) - 1)])
    time.sleep(1)