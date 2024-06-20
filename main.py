"""
Authors:
Otto Andrade -2190403
Brayam Fonseca
Daniela Cabrales
"""


import os
from Algorithm.img_processing import BackgroundRemover


def main():
    # Image route
    img_route = os.path.join(os.path.dirname(__file__),
                             'Algorithm/test_images',
                             'img19_retrato.jpeg')

    # Use example
    bg_remover = BackgroundRemover()
    bg_remover.load_image(img_route)
    bg_remover.show_image()
    bg_remover.apply_blur(blur_type='gaussian', kernel_size=15)
    bg_remover.show_blurred_image()

    return 0


if __name__ == '__main__':
    main()
