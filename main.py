import numpy as np
import cv2

from iter_methods import *


N = 15

# Test change

if __name__ == "__main__":
    img_objective = cv2.imread("circulos_coloridos.png")
    img_objective = (img_objective / 255)

    cv2.namedWindow("Objetivo Simple")
    cv2.moveWindow("Objetivo Simple", 500, 250);
    cv2.imshow("Objetivo Simple", img_objective)
    cv2.waitKey(0)

    x = simple_gradient(img_objective, MSE, N, lambda n: .3 / np.log(np.e + n), 1000, 1e-9, _delta=.05,
                        diff_scheme_to_use=2, use_threads=False, show_progress=True)
    quit()
