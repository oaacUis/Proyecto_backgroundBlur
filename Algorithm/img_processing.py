# Created by: Otto Andrade
# Created by: Brayan Fonseca

import cv2
import numpy as np
import matplotlib.pyplot as plt
from semanticSegmentationClass import DeepLabModel
from sklearn.cluster import KMeans
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
        g1_std = cv2.GaussianBlur(g1, (nn, nn), nn / 6,
                                  borderType=cv2.BORDER_REFLECT)
        # Gradiente en dirección x
        g1_gradient = np.abs(cv2.Sobel(g1, cv2.CV_64F, 1, 0, ksize=3))

        # Combinar los descriptores en un solo mapa de descriptores
        g = np.stack((g1_std, g1_gradient), axis=-1)

        # Reshape para que g sea del tamaño de f
        g = g.reshape(-1, g.shape[2])

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
                    mask = self.get_semantic_segmentation()
                elif method == "get_ORB_segmentation":
                    # mask = self.get_ORB_segmentation()
                    mask = np.zeros(shape=(self.image_shape[0],
                                           self.image_shape[1]))
                    
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
