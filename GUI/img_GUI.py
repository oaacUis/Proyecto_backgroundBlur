from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
import cv2


class ImageEditorApp(QWidget):
    def __init__(self):
        super().__init__()

        # Create a QVBoxLayout
        vbox = QVBoxLayout(self)

        # Create a QLabel for displaying the image
        self.label = QLabel(self)
        vbox.addWidget(self.label)

        # Create a QPushButton for loading an image and add it to the QVBoxLayout
        self.load_image_button = QPushButton("Load Image", self)
        self.load_image_button.clicked.connect(self.load_image)
        vbox.addWidget(self.load_image_button)

        # Create a QPushButton for applying the filter and add it to the QVBoxLayout
        self.apply_filter_button = QPushButton("Apply Gaussian Blur", self)
        self.apply_filter_button.clicked.connect(self.apply_filter)
        vbox.addWidget(self.apply_filter_button)

        # Set the QVBoxLayout as the layout for this QWidget
        self.setLayout(vbox)

    # def initUI(self):
    #    self.label_image = QLabel(self)
    #    self.label_image.setGeometry(50, 50, 500, 300)
    #    self.label_image.setScaledContents(True)

    #    btn_load_image = QPushButton("Cargar Imagen", self)
    #    btn_load_image.setGeometry(50, 10, 100, 30)
    #    btn_load_image.clicked.connect(self.load_image)

    def load_image(self):
        # Open a file dialog and get the path of the selected image file
        image_path, _ = QFileDialog.getOpenFileName()

        # Load the image using OpenCV
        self.cv_img = cv2.imread(image_path)

        # Convert the image from BGR color space to RGB color space
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)

        # Convert the OpenCV image to a PyQt5 QImage
        qt_img = QImage(self.cv_img.data, self.cv_img.shape[1], self.cv_img.shape[0], self.cv_img.strides[0], QImage.Format_RGB888)

        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(qt_img)

        # Set the QPixmap as the pixmap for the QLabel
        self.label.setPixmap(pixmap)
        
    def apply_filter(self):
        # Apply a Gaussian blur filter using OpenCV
        self.cv_img = cv2.GaussianBlur(self.cv_img, (15, 15), 0)

        # Convert the filtered image to a PyQt5 QImage
        qt_img = QImage(self.cv_img.data, self.cv_img.shape[1], self.cv_img.shape[0], self.cv_img.strides[0], QImage.Format_RGB888)

        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(qt_img)

        # Set the QPixmap as the pixmap for the QLabel
        self.label.setPixmap(pixmap)
