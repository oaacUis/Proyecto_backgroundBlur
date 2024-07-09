import os
import numpy as np
from img_processing import BackgroundRemover
import matplotlib.pyplot as plt

def main():
    # Image route
    image = 'img19_retrato.jpeg'
    #image = 'img12_urbano.jpeg'
    #image = 'img13_urbano.jpeg' # No funciona, cvt color 
    #image = 'img14_campo.jpeg' # No distorsiona el fondo
    #image = 'img15_campo.jpeg' #No funciona
    #image = 'img16_campo.jpeg' # No funciona, cvt color
    #image = 'img17_retrato.webp' #No distorsiona el fondo
    #image = 'img18_retrato.jpeg' #No funciona, todo distorsionado 
    #image = 'img20_retrato.jpeg' 
    img_route = os.path.join(os.path.dirname(__file__),
                             'test_images',
                             image)

    # Use example
    bg_remover = BackgroundRemover()
    bg_remover.load_image(img_route)
    # bg_remover.show_image(bg_remover.image_rgb)
    mask_list = {"get_semantic_segmentation": True, 
                 "get_texture_segmentation": False, "get_canny_segmentation": False, 
                 "get_sobel_segmentation": False, "get_hog_segmentation": False}
    bg_remover.get_final_mask(mask_dict=mask_list)
    
    # Assuming get_final_mask updates an attribute with the mask, e.g., bg_remover.segmentation_mask
    bg_remover.apply_final_mask()
    bg_remover.show_image(bg_remover.modified_image)
    
    print("--------Semantic---------")
    mask_used = bg_remover.get_semantic_segmentation()
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
    final_mask = bg_remover.normalized_mask
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
