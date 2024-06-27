import cv2
import numpy as np
from matplotlib import pyplot as plt

# Cargar la imagen de entrada
image_path = './img12_urbano.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Crear un detector ORB
orb = cv2.ORB_create()

# Detectar puntos clave y calcular descriptores
keypoints, descriptors = orb.detectAndCompute(image, None)

# Dibujar los puntos clave en la imagen
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

# Mostrar la imagen con los puntos clave
plt.imshow(image_with_keypoints, cmap='gray')
plt.title('Puntos Clave Detectados')
plt.show()