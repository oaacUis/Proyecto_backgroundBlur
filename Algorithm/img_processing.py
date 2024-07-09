# Created by: Otto Andrade
# Created by: Brayan Fonseca

import cv2
import numpy as np
import matplotlib.pyplot as plt
from semanticSegmentationClass import DeepLabModel
from sklearn.cluster import KMeans
from skimage.feature import hog
from scipy.signal import convolve2d
import os
import sys


class BackgroundRemover:
    def __init__(self):
        self.mask_list = []
        self.class_mask = None

    def load_image(self, image_path):
        self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_shape = self.image.shape

    def show_image(self, image, map=None):
        plt.imshow(image, cmap=map)
        plt.axis("off")
        plt.show()

    def apply_blur(self, objective, blur_type="gaussian", kernel_size=5):
        if blur_type == "gaussian":
            blurred_image = cv2.GaussianBlur(objective, (kernel_size, kernel_size), 0)
        elif blur_type == "median":
            blurred_image = cv2.medianBlur(objective, kernel_size)
        elif blur_type == "bilateral":
            blurred_image = cv2.bilateralFilter(
                objective, kernel_size, sigmaColor=75, sigmaSpace=75
            )
        return blurred_image

    def get_semantic_segmentation(self):
        """
        Performs semantic segmentation on the input image
        using a pre-trained DeepLab model.

        Returns:
            semanticMask_Resized (numpy.ndarray): The resized semantic
            segmentation mask.
        """
        ruta_modelo = "deeplab_model/"
        modelo = "deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb"
        full_model_path = os.path.join(os.path.dirname(__file__), ruta_modelo, modelo)

        # Check if the model path exists
        if not os.path.exists(full_model_path):
            print(f"Model file not found at {full_model_path}")
            sys.exit(1)  # Exit the script with an error status

        try:
            modelo_deeplab = DeepLabModel(full_model_path)
            # Continue with your code here
        except Exception as e:  # Consider catching more specific exceptions
            print(f"Failed to load the model: {e}")
            sys.exit(1)

        target_size = (513, 513)
        image_resized = cv2.resize(
            self.image, target_size, interpolation=cv2.INTER_LINEAR
        )
        imagen_array = np.copy(image_resized)

        resultado_prediccion = modelo_deeplab.run(imagen_array)
        img_result = resultado_prediccion.squeeze()
        mascara_prediccion = img_result.astype(np.float32)
        original_size = self.image.shape
        if len(original_size) == 3:
            prediction_size = (original_size[1], original_size[0])
        elif len(original_size) == 2:
            prediction_size = (original_size[1], original_size[0])
        semanticMask_Resized = cv2.resize(mascara_prediccion, prediction_size)
        print(type(semanticMask_Resized[0, 0]))
        # semanticMask_Resized = 1.0 - semanticMask_Resized  # Invert the mask
        return np.clip(semanticMask_Resized, a_min=0.0, a_max=1.0)

    def get_texture_segmentation(self):
        f = self.image_rgb
        f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        f = f.astype(np.float32) / 255.0

        # mean
        radius = 3
        kernel_mean = np.ones((radius, radius))
        kernel_mean = kernel_mean / kernel_mean.size
        mean = convolve2d(f, kernel_mean, mode="same")

        # variance
        sq_f = np.power(f, 2)
        sq_f = convolve2d(sq_f, kernel_mean, mode="same")
        sq_mean = np.power(mean, 2)
        variance = sq_f - sq_mean

        # deviation
        deviation = np.sqrt(np.abs(variance))
        deviation = deviation / np.max(np.abs(deviation))

        mask = deviation
        th_s = 0.1
        s_umbralizada = mask > th_s

        # Define el elemento estructurante para la dilatación
        kernel = np.ones(
            (3, 3), np.uint8
        )  # Puedes ajustar el tamaño del kernel según sea necesario
        # Aplica la dilatación
        a = ~s_umbralizada
        # eroded_mask = cv2.erode(a, kernel, iterations=1)
        # dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)  # Puedes ajustar el número de iteraciones

        return a
    def get_canny_segmentation(self):
        image = self.image_rgb
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.uint8)

        # Aplicar el detector de bordes de Canny
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)

        # Crear una máscara binaria a partir de los bordes detectados
        mask = edges
        mask[mask > 0] = 255
        th_s = 0.1
        mask[mask > th_s] = 1
        mask[mask <= th_s] = 0
        s_umbralizada = mask > th_s
        a = ~s_umbralizada

        return mask
    def get_hog_segmentation(image_path):
        """
        This function segments an image using HOG (Histogram of Oriented Gradients)
        and returns a binary mask.

        Args:
            image_path (str): Path to the image file.
            threshold (float, optional): Threshold value for binarization. Defaults to 0.1.

        Returns:
            numpy.ndarray: A binary mask representing the segmented image.
        """
        # Read image
        img = image_path

        # Calculate HOG features
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, multichannel=True)

        # Apply thresholding for binarization
        threshold=20
        mask = hog_image
        mask[mask < threshold] = 0
        mask[mask >= threshold] = 255

        return mask
    def get_sobel_segmentation(image):
        # Convertir la imagen a escala de grises
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aplicar un filtro de suavizado (opcional)
        image = cv2.GaussianBlur(image, (5, 5), 0)

        # Normalizar la imagen
        image = image.astype(np.float32) / 255.0

        # Calcular los gradientes Sobel
        sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)

        # Calcular la magnitud del gradiente
        sobelCombined = np.sqrt(sobelX**2 + sobelY**2)

        # Normalizar y convertir a uint8
        sobelCombined = np.uint8(255 * sobelCombined / np.max(sobelCombined))

        # Aplicar un umbral para crear una máscara binaria
        _, mask = cv2.threshold(sobelCombined, 50, 255, cv2.THRESH_BINARY)

        return mask

    def get_final_mask(self, mask_dict: dict):
        """_summary_

        Args:
            mask_dict (dict): It's a dictionary with key:value pairs
            where key is the name of the mask and value is a bool value
            if is going to be used or not. ex:
            mask_list = {   "get_semantic_segmentation": True,
                            "get_ORB_segmentation": False, ...}

        Returns:
            np.array: Final mask after Kmeans algorithm
        """
        m, n, _ = self.image.shape

        for method, use_mask in mask_dict.items():
            if use_mask:
                if method == "get_semantic_segmentation":
                    mask = self.get_semantic_segmentation().astype(np.float32)
                elif method == "get_ORB_segmentation":
                    # mask = self.get_ORB_segmentation()
                    mask = np.zeros(shape=(self.image_shape[0],
                                           self.image_shape[1]))
                elif method == "get_texture_segmentation":
                    mask = self.get_texture_segmentation() * 0.1
                elif method == "get_canny_segmentation":
                    mask = self.get_canny_segmentation() *0.1
                elif method == "get_hog_segmentation":
                    mask = self.get_hog_segmentation() 
                elif method == "get_sobel_segmentation":
                    mask = self.get_sobel_segmentation(self.image) 
                self.mask_list.append(mask.reshape(m * n, 1))

        X = np.hstack(tuple(self.mask_list))
        print("Kmeans shape: ", X.shape)
        final_mask = KMeans(n_clusters=2, n_init="auto").fit(X)
        # centers = final_mask.cluster_centers_
        labels = final_mask.labels_

        # Verify if the most common class is 0
        # unique, counts = np.unique(labels, return_counts=True)
        # most_common_class = unique[np.argmax(counts)]
        if labels[0] != 0:
            # Intercambiar etiquetas
            labels = np.where(labels == 0, 1, 0)

        self.class_mask = labels.reshape(m, n)
        # return class_mask

    def apply_final_mask(self, blur_type="gaussian", kernel_size=5):
        """
        Applies the final mask to the image.

        Args:
            blur_type (str, optional): The type of blur to apply.
            Defaults to "gaussian".
            kernel_size (int, optional): The size of the kernel for
            blurring. Defaults to 5.

        Returns:
            None
        """
        # mask = np.dstack((self.class_mask, self.class_mask, self.class_mask))
        blurred_background = cv2.GaussianBlur(
            self.image_rgb, (kernel_size, kernel_size), cv2.BORDER_DEFAULT
        )
        masked_image = imblend(
            self.image_rgb.astype(np.float32) / 255.0,
            blurred_background.astype(np.float32) / 255.0,
            np.clip(self.class_mask.astype(np.float32), a_min=0.0, a_max=1.0),
            nlevels=5,
        )*255
        # masked_image = blurred_background * mask + self.image_rgb *(1 - mask)
        self.modified_image = np.copy(masked_image.astype(np.uint8))
        # print(self.modified_image.shape)
