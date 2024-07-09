# Created by: Otto Andrade
# Created by: Brayan Fonseca

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.feature import hog
from scipy.signal import convolve2d
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from semanticSegmentationClass import DeepLabModel


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

        return semanticMask_Resized.astype(np.float32)

    def get_texture_segmentation(self):
        """
        Performs texture segmentation on the input image by calculating
        the local deviation from the mean pixel value.

        Returns:
            numpy.ndarray: The binary segmentation mask.
        """
        # Convert the input RGB image to grayscale
        f = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Normalize the grayscale image to the range [0, 1]
        f = f.astype(np.float32) / 255.0
        
        # Define the kernel for the mean filter
        radius = 3
        kernel_mean = np.ones((radius, radius)) / (radius * radius)
        
        # Apply mean filter to the image
        mean = convolve2d(f, kernel_mean, mode="same")
        
        # Calculate the variance
        sq_f = convolve2d(np.power(f, 2), kernel_mean, mode="same")
        sq_mean = np.power(mean, 2)
        variance = sq_f - sq_mean
        
        # Calculate the deviation
        deviation = np.sqrt(np.abs(variance))
        deviation = deviation / np.max(np.abs(deviation))
        
        # Create binary mask by thresholding the deviation
        th_s = 0.1
        s_umbralizada = deviation > th_s
        
        # Invert the binary mask
        a = ~s_umbralizada

        return a
    
    def get_canny_segmentation(self):
        """
        Performs edge detection segmentation on the input image using the Canny edge detector.

        Returns:
            numpy.ndarray: The binary segmentation mask.
        """
        # Convert the input RGB image to grayscale
        image = self.image_rgb
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.uint8)

        # Apply the Canny edge detector
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)

        # Create a binary mask from the detected edges
        mask = np.zeros_like(edges)
        mask[edges > 0] = 1

        return mask
    
    def get_hog_segmentation(self):
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
        img = self.image_rgb

        # Calculate HOG features
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, channel_axis=-1)

        # Apply thresholding for binarization
        threshold=20
        mask = hog_image
        mask[mask < threshold] = 0
        mask[mask >= threshold] = 255

        return mask
    
    def get_sobel_segmentation(self):
        """
        Performs edge detection segmentation on the input image using the Sobel operator.

        Args:
            image (numpy.ndarray): Input image in RGB format.

        Returns:
            numpy.ndarray: The binary segmentation mask.
        """
        # Convert the input image to grayscale
        image = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)

        # Apply a Gaussian blur to the image to reduce noise (optional)
        image = cv2.GaussianBlur(image, (5, 5), 0)

        # Normalize the image to the range [0, 1]
        image = image.astype(np.float32) / 255.0

        # Calculate the Sobel gradients in the X and Y directions
        sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)

        # Calculate the gradient magnitude
        sobelCombined = np.sqrt(sobelX**2 + sobelY**2)

        # Normalize the gradient magnitude and convert it to uint8
        sobelCombined = np.uint8(255 * sobelCombined / np.max(sobelCombined))

        # Apply a threshold to create a binary mask
        _, mask = cv2.threshold(sobelCombined, 50, 255, cv2.THRESH_BINARY)

        return mask
    
    def get_final_mask(self, mask_dict: dict):
        """
        Combines multiple segmentation masks using KMeans clustering.

        Args:
            mask_dict (dict): A dictionary where the key is the mask name
                            and the value is a bool indicating whether
                            the mask should be used.
                            Example: {"get_semantic_segmentation": True, "get_ORB_segmentation": False, ...}

        Returns:
            np.array: Final binary mask after KMeans clustering.
        """
        m, n, o = self.image.shape
        self.mask_list = []

        for method, use_mask in mask_dict.items():
            if use_mask:
                if method == "get_semantic_segmentation":
                    mask = self.get_semantic_segmentation()
                elif method == "get_ORB_segmentation":
                    # mask = self.get_ORB_segmentation()
                    mask = np.zeros(shape=(self.image_shape[0], self.image_shape[1]))
                elif method == "get_texture_segmentation":
                    mask = self.get_texture_segmentation() * 0.1
                elif method == "get_canny_segmentation":
                    mask = self.get_canny_segmentation() * 0.1
                elif method == "get_hog_segmentation":
                    mask = self.get_hog_segmentation() * 0.001
                elif method == "get_sobel_segmentation":
                    mask = self.get_sobel_segmentation() * 0.0025

                self.mask_list.append(mask.reshape(m * n, 1))

        # Stack all masks horizontally
        X = np.hstack(tuple(self.mask_list))
        print("KMeans shape: ", X.shape)

        # Apply KMeans clustering
        final_mask = KMeans(n_clusters=2, n_init="auto").fit(X)
        labels = final_mask.labels_

        # Verify if the most common class corresponds to the background or the object
        if np.sum(labels) > len(labels) / 2:
            # If the majority is foreground, invert labels
            labels = 1 - labels

        self.class_mask = ~labels.reshape(m, n)
        return self.class_mask
    
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

        Note:
            Mask attrib spected to be a numpy array from 0.0 to 1.0
        """
        mask = np.dstack((self.class_mask, self.class_mask, self.class_mask))
        blurred_background = cv2.GaussianBlur(
            self.image_rgb, (kernel_size, kernel_size), cv2.BORDER_DEFAULT
        )
        # print("Mask type: ", type(mask))
        # print("background type: ", type(blurred_background))
        masked_image = blurred_background * mask + self.image_rgb * (1 - mask)
        self.modified_image = np.copy(masked_image)
        # print(self.modified_image.shape)

"""
Tareas Pendientess:
- Estandarizar las salidas de las funciones para obtener máscaras
- Comprobar el dato de entrada y salida del kmeans
- Testear las mascaras de segmentación y la mascara final
- Reimplementar el bending usando piramides
- Recordar que el class_mask debe ser un numpy array de tipo float32 con valores 0.0 a 1.0
"""