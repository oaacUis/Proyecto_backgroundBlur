"""
Authors:
Otto Andrade -2190403
Brayam Fonseca
Daniela Cabrales
"""

import os
import sys
#from Algorithm.img_processing import BackgroundRemover
from img_GUI import ImageEditorApp
from PyQt5.QtWidgets import QApplication


def main():
    # Image route
    # img_route = os.path.join(os.path.dirname(__file__),
    #                          'Algorithm/test_images',
    #                          'img19_retrato.jpeg')

    app = QApplication(sys.argv)
    window = ImageEditorApp()
    window.show()
    sys.exit(app.exec_())
    return 0


if __name__ == '__main__':
    main()
