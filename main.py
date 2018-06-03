import numpy as np
import cv2
from iter_methods import *
from measures import *

N = 5

if __name__ == "__main__":
    img_objective = cv2.imread("Test_cases/blue_circle.png")
    img_objective = (img_objective / 255)
    np.random.seed(1235)
    cv2.namedWindow("Objetivo Simple")
    cv2.moveWindow("Objetivo Simple", 500, 250)
    cv2.imshow("Objetivo Simple", img_objective)
    cv2.waitKey(0)

    x = simple_gradient_method_parallel(img_objective, N, -3, lambda n: .3 / np.log(np.e + n), 2000, 1e-4, _delta=.1,
                                        diff_scheme_to_use=2, show_progress=True)
    quit()
