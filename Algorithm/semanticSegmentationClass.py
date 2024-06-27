import tensorflow as tf
from PIL import Image


class DeepLabModel:
    def __init__(self, model_path):
        # Cargar el grafo desde el archivo .pb
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, "rb") as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")

        # Obtener el tensor de entrada y salida del grafo
        self.input_tensor = self.graph.get_tensor_by_name("ImageTensor:0")
        self.output_tensor = self.graph.get_tensor_by_name(
            "SemanticPredictions:0"
        )

    def run(self, image_array):
        # Crear una sesión TensorFlow
        with tf.compat.v1.Session(graph=self.graph) as sess:
            # Ejecutar la predicción
            predicted_map = sess.run(
                self.output_tensor,
                feed_dict={self.input_tensor: [image_array]}
            )
        return predicted_map > 0.1


# Función para redimensionar la imagen
def resize_image(image, target_size):
    return image.resize(target_size, Image.NEAREST)


# Función para redimensionar la máscara
def resize_mask(mask, target_size):
    return mask.resize(target_size, Image.NEAREST)
