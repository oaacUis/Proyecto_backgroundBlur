from PyQt5.QtWidgets import QMainWindow, QLabel, QFileDialog, QAction
from PyQt5.QtWidgets import QToolBar, QCheckBox, QGroupBox, QVBoxLayout
from PyQt5.QtWidgets import QWidget, QPushButton, QProgressBar, QInputDialog
from PyQt5.QtWidgets import QMessageBox, QColorDialog, QSplashScreen
from PyQt5.QtWidgets import QHBoxLayout, QLineEdit
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from Algorithm.img_processing import BackgroundRemover
import cv2
import sys
import time
import numpy as np
import matplotlib.pyplot as plt


class ImageEditorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GIPFOCUS")
        self.setGeometry(100, 100, 800, 800)

        # Cargar y redimensionar el logo
        logo_path = "./icons/logo.png"
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

        # From BackgroundRemover
        self.bg_remover = BackgroundRemover()
        self.mask_list = {
            "get_semantic_segmentation": False,
            "get_texture_segmentation": False,
            "get_canny_segmentation": False,
            "get_sobel_segmentation": False,
            "get_hog_segmentation": False,
        }

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

        eraser_button = QAction(QIcon(), "Borrador", self)
        eraser_button.triggered.connect(self.use_eraser)
        toolbar_top.addAction(eraser_button)

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

        self.checkboxes = []

        # Estructura definir los filtros
        
        # Filtro 1: Segmentación semántica
        self.semanticSegmentationCheckBox = QCheckBox("Semantic Segmentation", self)
        option_layout.addWidget(self.semanticSegmentationCheckBox)
        
        # Filtro 2: Segmentación de texturas
        self.textureSegmentationCheckBox = QCheckBox("Texture Segmentation", self)
        option_layout.addWidget(self.textureSegmentationCheckBox)

        # Filtro 3: Segmentación de bordes con Canny
        self.cannySegmentationCheckBox = QCheckBox("Canny Segmentation", self)
        option_layout.addWidget(self.cannySegmentationCheckBox)

        # Filtro 4: Segmentación de bordes con Sobel
        self.sobelSegmentationCheckBox = QCheckBox("Sobel Segmentation", self)
        option_layout.addWidget(self.sobelSegmentationCheckBox)

        # Filtro 5: Segmentación de bordes con HOG
        self.hogSegmentationCheckBox = QCheckBox("HOG Segmentation", self)
        option_layout.addWidget(self.hogSegmentationCheckBox)

        # Add more checkboxes as needed ...

        # Connect checkboxes to their respective slot functions
        self.semanticSegmentationCheckBox.stateChanged.connect(self.updateMaskDict)
        self.textureSegmentationCheckBox.stateChanged.connect(self.updateMaskDict)
        self.cannySegmentationCheckBox.stateChanged.connect(self.updateMaskDict)
        self.sobelSegmentationCheckBox.stateChanged.connect(self.updateMaskDict)
        self.hogSegmentationCheckBox.stateChanged.connect(self.updateMaskDict)

        # Initialize the QLineEdit for Gaussian Blur value input
        self.gaussianBlurValueInput = QLineEdit(self)
        self.gaussianBlurValueInput.setPlaceholderText("Gaussian Blur Value")
        
        self.checkGaussianBlurValueButton = QPushButton("Set", self)
        self.checkGaussianBlurValueButton.clicked.connect(self.checkGaussianBlurValue)
        
        self.gaussianBlurLayout = QHBoxLayout()
        self.gaussianBlurLayout.addWidget(self.gaussianBlurValueInput)
        self.gaussianBlurLayout.addWidget(self.checkGaussianBlurValueButton)
        option_layout.addLayout(self.gaussianBlurLayout)
        
        # side_layout.addWidget(apply_button)

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

    def updateMaskDict(self):
        # Update the mask dictionary based on the checkboxes
        self.mask_list["get_semantic_segmentation"] = (
            self.semanticSegmentationCheckBox.isChecked()
        )
        self.mask_list["get_texture_segmentation"] = (
            self.textureSegmentationCheckBox.isChecked()
        )
        self.mask_list["get_canny_segmentation"] = (
            self.cannySegmentationCheckBox.isChecked()
        )
        self.mask_list["get_sobel_segmentation"] = (
            self.sobelSegmentationCheckBox.isChecked()
        )
        self.mask_list["get_hog_segmentation"] = (
            self.hogSegmentationCheckBox.isChecked()
        )
        # Update the other checkboxes as needed
        print("Mask list: ", self.mask_list)

    # Métodos para manejar las acciones de los botones "Aplicar" y "Visualizar"
    def apply_filters(self):
        print("Applying filters...")
        # self.bg_remover.get_final_mask(mask_dict=self.mask_list)
        self.bg_remover.apply_final_mask()
        print("Final mask already obtained")
        k = self.bg_remover.modified_image*255
        k = k.astype(np.uint8)
        print("Result mask shape from apply", k.shape)
        print(f"Max value in pixmap_final: {np.max(k)}")
        print(f"Min value in pixmap_final: {np.min(k)}")
        print(f"Type of pixmap_final: {type(k[0, 0, 0])}")
        self.pixmap_final = self.convert_cv_qt(k, saveSize=True)
        self.label_image.setPixmap(self.pixmap_final)

    def visualize_filters(self):
        print("Visualizing current mask from filters")
        
        self.bg_remover.get_final_mask(mask_dict=self.mask_list)
        self.result_mask = self.bg_remover.class_mask * 255
        self.result_mask = self.result_mask.astype(np.uint8)
        print("Result mask shape from visualize", self.result_mask.shape)
        print(f"Max value in final_mask: {np.max(self.result_mask)}")
        print(f"Min value in final_mask: {np.min(self.result_mask)}")
        print(f"Type of final_mask: {type(self.result_mask[0, 0])}")
        self.pixmap_mask = self.convert_cv_qt(
            self.result_mask, saveSize=True,
            setResize=True, objetiveSize=(600, 300))

        # canvas = QPixmap(600, 300)  # Adjust size as needed
        # canvas.fill(Qt.red)
        self.preview_label.setPixmap(self.pixmap_mask)

    def checkGaussianBlurValue(self):
        value = self.gaussianBlurValueInput.text()
        try:
            value = int(value)
            if value <= 0:
                raise ValueError("The value must be positive")
            # Value is valid, you can use it for setting the Gaussian Blur value
            print(f"Valid Gaussian Blur value: {value}")
            self.bg_remover.set_GaussianBlurValue(value)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Value", str(e))

    def load_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Archivos de Imagen (*.jpg *.jpeg *.png)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            file_name = file_dialog.selectedFiles()[0]
            # pixmap = QPixmap(file_name)
            self.bg_remover.load_image(file_name)
            pixmap = self.convert_cv_qt(file_name, saveSize=True)
            self.label_image.setPixmap(pixmap)
            print("New image to: ", self.label_image.pixmap())
            # Esto no funciona adecuadamente
            # self.label_image.setScaledContents(True)
            # print("New image to (After scale): ", self.label_image.pixmap())
            self.reset_meths()

    def save_image(self):
        pixmap = self.label_image.pixmap()
        # pixmap = self.pixmap_mask
        #pixmap = self.pixmap_final
        if pixmap:
            save_dialog = QFileDialog(self)
            save_dialog.setAcceptMode(QFileDialog.AcceptSave)
            save_dialog.setNameFilter("Archivos de Imagen (*.jpg *.jpeg *.png)")
            save_dialog.setDefaultSuffix("png")

            if save_dialog.exec_():
                save_path = save_dialog.selectedFiles()[0]
                # Opcion 1 - Guardar con el tamaño original
                n, m, _ = self.bg_remover.image_shape
                a = self.convert_qt_cv(pixmap, setResize=True)
                pixmap = self.convert_cv_qt(a, setResize=False,
                                            objetiveSize=(n, m))

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
        if color.isValid() and self.currentTool != "eraser":
            self.brush_color = color

    def select_brush_size(self):
        size, ok = QInputDialog.getInt(
            self,
            "Seleccionar Tamaño del Pincel",
            "Tamaño del Pincel:",
            self.brush_size,
            1,
            50,
            1,
        )
        if ok:
            self.brush_size = size

    def use_pencil(self):
        if self.currentTool == "pencil":
            self.currentTool = None
        else:
            self.currentTool = "pencil"
            self.select_brush_size()
            print("Current tool: Pencil")

    def use_eraser(self):
        if self.currentTool == "eraser":
            self.currentTool = None
        else:
            self.currentTool = "eraser"
            self.brush_color = Qt.white  # Assuming white background
            self.select_brush_size()
            print("Current tool: Eraser")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.currentTool == "pencil":
            print("Mouse pressed")
            self.drawing = True
            # Convert global position to local position relative to label_image
            localPos = self.label_image.mapFromGlobal(event.globalPos())
            self.lastPoint = localPos
        
        if event.button() == Qt.LeftButton and self.currentTool == "eraser":
            print("Mouse pressed")
            self.drawing = True
            # Convert global position to local position relative to label_image
            localPos = self.label_image.mapFromGlobal(event.globalPos())
            self.lastPoint = localPos

    def mouseMoveEvent(self, event):
        if (
            (event.buttons() & Qt.LeftButton)
            and self.drawing
            and (self.currentTool == "pencil" or self.currentTool == "eraser")
        ):
            print("Mouse is moving...")
            pixmap = self.label_image.pixmap()
            print(self.label_image.pixmap())
            painter = QPainter(pixmap)
            painter.setPen(
                QPen(
                    self.brush_color,
                    self.brush_size,
                    Qt.SolidLine,
                    Qt.RoundCap,
                    Qt.RoundJoin,
                )
            )
            # Convert global position to local position relative to label_image
            localPos = self.label_image.mapFromGlobal(event.globalPos())
            painter.drawLine(self.lastPoint, localPos)
            self.lastPoint = localPos
            self.label_image.setPixmap(pixmap)

    def mouseReleaseEvent(self, event):
        if (
            event.button() == Qt.LeftButton
            and self.drawing
            and (self.currentTool == "pencil" or self.currentTool == "eraser")
        ):
            print("Mouse released")
            self.drawing = False
            # self.label_image.setPixmap(self.label_image.pixmap())
            # self.counter += 1
            print("Counter: ", self.counter)
            # if self.counter > 3:
            #     self.save_image()

    # To convert from opencv to QPixmap
    def convert_cv_qt(
        self, cv_img, saveSize=False, setResize=True, objetiveSize=(700, 600)
    ):
        """Convert from an opencv image to QPixmap"""
        if isinstance(cv_img, np.ndarray):
            # cv_img is a numpy array
            self.rgb_image = cv_img.astype(np.uint8)
            if len(self.rgb_image.shape) == 2:
                self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_GRAY2RGB)

        else:
            image = cv2.imread(cv_img)
            self.rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if saveSize:  # Save the original size of the image
            self.original_image_shape = self.rgb_image.shape

        if setResize:  # Resize the image to a fixed size
            self.rgb_image = cv2.resize(self.rgb_image, objetiveSize)

        h, w, ch = self.rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            self.rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        # En caso de querer escalar la imagen QT
        # p = convert_to_Qt_format.scaled(700, 600, Qt.KeepAspectRatio)
        p = convert_to_Qt_format
        return QPixmap.fromImage(p)

    def convert_qt_cv(
        self, qt_img, setResize=False
    ):  # To convert from QPixmap to opencv
        """Convert from a QPixmap to an opencv image"""
        qimage = qt_img.toImage()
        if qimage.format() != QImage.Format_RGB888:
            qimage = qimage.convertToFormat(QImage.Format_RGB888)
        b = qimage.bits()
        b.setsize(qimage.byteCount())
        stride = qimage.bytesPerLine()
        # cv_img = np.array(b).reshape(qimage.height(), qimage.width(), 3)
        cv_img = np.frombuffer(b, dtype=np.uint8).reshape(
            (qimage.height(), stride // 3, 3)
        )
        if setResize:
            cv_img = cv2.resize(
                cv_img, (self.original_image_shape[1], self.original_image_shape[0])
            )
        return cv_img
    
    def convert_grayscale_qt(self, grayscale_image, saveSize=False,
                             setResize=True, objetiveSize=(600, 300)):
        
        self.grayscale_img = grayscale_image
        if isinstance(grayscale_image, str):  # If the input is a file path
            self.grayscale_img = cv2.imread(grayscale_image, cv2.IMREAD_GRAYSCALE)
        elif not isinstance(grayscale_image, np.ndarray):
            raise ValueError("The input must be a file path or a numpy array.")

        # Ensure the image is in grayscale format (2D numpy array)
        if len(grayscale_image.shape) != 2:
            raise ValueError("The input image is not in grayscale format.")

        # Convert the grayscale image to QPixmap using the existing method
        self.gray_height, self.gray_width = self.grayscale_img.shape
        bytesPerLine = self.gray_width

        if saveSize:  # Save the original size of the image
            self.original_grayimage_shape = (self.gray_height, self.gray_width)

        if setResize:  # Resize the image to a fixed size
            self.grayscale_img = cv2.resize(self.grayscale_img, objetiveSize)
        plt.imshow(self.grayscale_img, cmap='gray')
        # Convert the numpy array to QImage
        qImg = QImage(self.grayscale_img.data, self.gray_width, self.gray_height,
                      bytesPerLine, QImage.Format_Grayscale8)

        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(qImg)
        
        return pixmap


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
