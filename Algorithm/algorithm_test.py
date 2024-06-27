import os
from img_processing import BackgroundRemover

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
    mask_list = {"get_semantic_segmentation": True}
    bg_remover.get_final_mask(mask_dict=mask_list)
    bg_remover.apply_final_mask()
    bg_remover.show_image(bg_remover.modified_image)

    return 0


if __name__ == '__main__':
    main()
