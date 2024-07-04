# Created by: Otto Andrade
# Created by: Brayan Fonseca

import cv2
import numpy as np
import matplotlib.pyplot as plt
from semanticSegmentationClass import DeepLabModel
from sklearn.cluster import KMeans
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

    def show_image(self, image):
        plt.imshow(image)
        plt.axis("off")
        plt.show()

    def apply_blur(self, objective, blur_type="gaussian", kernel_size=5):
        if blur_type == "gaussian":
            blurred_image = cv2.GaussianBlur(objective,
                                             (kernel_size, kernel_size), 0)
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
        full_model_path = os.path.join(os.path.dirname(__file__),
                                       ruta_modelo,
                                       modelo)

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
        mascara_prediccion = img_result.astype(np.uint8)
        original_size = self.image.shape
        if len(original_size) == 3:
            prediction_size = (original_size[1], original_size[0])
        elif len(original_size) == 2:
            prediction_size = (original_size[1], original_size[0])
        semanticMask_Resized = cv2.resize(mascara_prediccion, prediction_size)

        return semanticMask_Resized

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
        kernel = np.ones((3, 3), np.uint8)  # Puedes ajustar el tamaño del kernel según sea necesario
        # Aplica la dilatación
        a = ~s_umbralizada
        #eroded_mask = cv2.erode(a, kernel, iterations=1)
        #dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)  # Puedes ajustar el número de iteraciones

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
        s_umbralizada = mask>th_s
        a = ~s_umbralizada

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
        m, n, o = self.image.shape

        for method, use_mask in mask_dict.items():
            if use_mask:
                if method == "get_semantic_segmentation":
                    mask = self.get_semantic_segmentation().astype(np.float32)
                elif method == "get_ORB_segmentation":
                    # mask = self.get_ORB_segmentation()
                    mask = np.zeros(shape=(self.image_shape[0],
                                           self.image_shape[1]))
                elif method == "get_texture_segmentation":
                    mask = self.get_texture_segmentation() *0.1
                elif method == "get_canny_segmentation":
                    mask = self.get_canny_segmentation() *0.1
                self.mask_list.append(mask.reshape(m * n, 1))

        X = np.hstack(tuple(self.mask_list))
        print("Kmeans shape: ", X.shape)
        final_mask = KMeans(n_clusters=2, n_init="auto").fit(X)
        # centers = final_mask.cluster_centers_
        labels = final_mask.labels_

        # Verificar si la clase mas común corresponde al fondo o al objeto
        if labels[-1] == 0:
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
        mask = np.dstack((self.class_mask, self.class_mask, self.class_mask))
        blurred_background = cv2.GaussianBlur(
            self.image_rgb, (kernel_size, kernel_size), cv2.BORDER_DEFAULT
        )
        masked_image = blurred_background * mask + self.image_rgb * (1 - mask)
        self.modified_image = np.copy(masked_image)
        # print(self.modified_image.shape)
