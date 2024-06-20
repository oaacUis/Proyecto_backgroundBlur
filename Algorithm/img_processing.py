# Created by: Otto Andrade
# Created by: Brayan Fonseca

import cv2
import numpy as np
import matplotlib.pyplot as plt


class BackgroundRemover:
    def __init__(self):
        pass

    def load_image(self, image_path):
        self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def show_image(self):
        plt.imshow(self.image_rgb)
        plt.axis('off')
        plt.show()

    def apply_blur(self, blur_type='gaussian', kernel_size=5):
        if blur_type == 'gaussian':
            self.blurred_image = cv2.GaussianBlur(self.image,
                                                  (kernel_size, kernel_size),
                                                  0)
        elif blur_type == 'median':
            self.blurred_image = cv2.medianBlur(self.image, kernel_size)
        elif blur_type == 'bilateral':
            self.blurred_image = cv2.bilateralFilter(self.image,
                                                     kernel_size,
                                                     sigmaColor=75,
                                                     sigmaSpace=75)

    def show_blurred_image(self):
        blurred_image_rgb = cv2.cvtColor(self.blurred_image, cv2.COLOR_BGR2RGB)
        plt.imshow(blurred_image_rgb)
        plt.axis('off')
        plt.show()

    def media(self, k):
        """f es la imagen y k el tamaño del kernel
        f: puede ser una imagen a color
        nn:
        Tamaño del vecindario para la desviación estándar local
        """
        nn = 15  
        # Convertir a escala de grises
        g1 = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)  
        # Suavizado con filtro gaussiano
        g1_std = cv2.GaussianBlur(g1, (nn, nn), nn/6,
                                  borderType=cv2.BORDER_REFLECT)  
        # Gradiente en dirección x
        g1_gradient = np.abs(cv2.Sobel(g1, cv2.CV_64F, 1, 0, ksize=3))  

        # Combinar los descriptores en un solo mapa de descriptores
        g = np.stack((g1_std, g1_gradient), axis=-1) 

        # Reshape para que g sea del tamaño de f
        g = g.reshape(-1, g.shape[2])

    # Lets add here some segmentation techniques
