import os
from img_processing import BackgroundRemover
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    # Image route
    image = 'img19_retrato.jpeg'
    img_route = os.path.join(os.path.dirname(__file__),
                             'test_images',
                             image)

    # Use example
    bg_remover = BackgroundRemover()
    bg_remover.load_image(img_route)
    # bg_remover.show_image(bg_remover.image_rgb)
    mask_list = {"get_semantic_segmentation": True, "get_texture_segmentation": True, "get_canny_segmentation": True}
    bg_remover.get_final_mask(mask_dict=mask_list)
    
    # Assuming get_final_mask updates an attribute with the mask, e.g., bg_remover.segmentation_mask
    bg_remover.apply_final_mask()
    bg_remover.show_image(bg_remover.modified_image)
    
    # Show the segmentation mask
    # This assumes there's a method show_image in BackgroundRemover and an attribute for the mask
    bg_remover.show_image(bg_remover.get_semantic_segmentation())
    bg_remover.show_image(bg_remover.get_canny_segmentation())
    # Obtener la máscara de segmentación semántica
    canny_mask = bg_remover.get_canny_segmentation()
    final_mask = bg_remover.class_mask
    # Graficar la máscara de segmentación semántica
    plt.imshow(final_mask, cmap='gray')  # Asumiendo que es una imagen en escala de grises
    plt.title('Máscara final aplicada')
    plt.axis('off')  # Omitir los ejes para una visualización más limpia
    plt.show()

    return 0

if __name__ == '__main__':
    main()
