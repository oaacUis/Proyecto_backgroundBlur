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
            self.blurred_image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        elif blur_type == 'median':
            self.blurred_image = cv2.medianBlur(self.image, kernel_size)
        elif blur_type == 'bilateral':
            self.blurred_image = cv2.bilateralFilter(self.image, kernel_size, sigmaColor=75, sigmaSpace=75)

    def show_blurred_image(self):
        blurred_image_rgb = cv2.cvtColor(self.blurred_image, cv2.COLOR_BGR2RGB)
        plt.imshow(blurred_image_rgb)
        plt.axis('off')
        plt.show()

    # Lets add here some segmentation techniques
