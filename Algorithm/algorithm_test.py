import os
from img_processing import BackgroundRemover
import matplotlib.pyplot as plt
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def main():
    # Image route
    #image = "Brayan_8.jpg"
    image = "img20_retrato.jpeg"
    img_route = os.path.join(os.path.dirname(__file__), "test_images", image)

    # Use example
    bg_remover = BackgroundRemover()
    bg_remover.load_image(img_route)
    # bg_remover.show_image(bg_remover.image_rgb)
    mask_list = {
        "get_semantic_segmentation": True,
        "get_texture_segmentation": True,
        "get_canny_segmentation": True,
        "get_sobel_segmentation": True,
        "get_hog_segmentation": True,
    }
    bg_remover.get_final_mask(mask_dict=mask_list)
    
    # Assuming get_final_mask updates an attribute with the mask, e.g.,
    bg_remover.apply_final_mask()
    plt.imshow(bg_remover.modified_image),plt.axis("off"), plt.title("Imagen final"),plt.show()

    print("--------HOG---------")
    mask_used = bg_remover.get_canny_segmentation()
    print(mask_used.shape)
    if mask_used.ndim == 2:
        print(type(mask_used[0,0]))
    else:
        print(type(mask_used[0,0,0]))
    print(type(mask_used))
    
    print(f"Max value in mask: {np.max(mask_used)}")
    print(f"Min value in mask: {np.min(mask_used)}")
    
    # Show the segmentation mask
    # This assumes there's a method show_image in BackgroundRemover and an attribute for the mask
    #bg_remover.show_image(bg_remover.get_semantic_segmentation())
    #bg_remover.show_image(bg_remover.get_sobel_segmentation())
    # Obtener la máscara de segmentación semántica
    # canny_mask = bg_remover.get_canny_segmentation()
    # sobel_mask = bg_remover.get_sobel_segmentation()
    # hog_mask = bg_remover.get_hog_segmentation()
    print("--------Final mask---------")
    final_mask = bg_remover.get_canny_segmentation()
    print(final_mask.shape)
    if final_mask.ndim == 2:
        print(type(final_mask[0,0]))
    else:
        print(type(final_mask[0,0,0]))
    print(type(final_mask))
    
    print(f"Max value in final_mask: {np.max(final_mask)}")
    print(f"Min value in final_mask: {np.min(final_mask)}")
    
    
    # Graficar la máscara de segmentación semántica
    plt.imshow(final_mask, cmap='gray')  # Asumiendo que es una imagen en escala de grises
    plt.title('Máscara aplicada')
    plt.axis('off')  # Omitir los ejes para una visualización más limpia
    plt.show()


    return 0


if __name__ == '__main__':
    main()
