import numpy as np
import cv2
import random
from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_mse as MSE
import multiprocessing as mp
import time
from scipy.ndimage.measurements import center_of_mass
import pickle
import matplotlib.pyplot as plt

np.random.seed(61)
H = 37
W = 27
N = 8


def triag_matrix(x, alpha):
    """
    Draws the shape of a triangle in a zero matrix with the value of alpha
    :param x: 3,2 array, the coords of the vertex
    :param alpha: double, opacity
    :return: H,W array, shape matrix
    """
    matrix = np.zeros((H, W), np.dtype('double'))
    if x.shape != (3, 2):
        x.shape = 3, 2
    x = x.astype(int)
    cv2.fillConvexPoly(matrix, x, alpha)
    return matrix


def circle_matrix(x, r, alpha):
    """
     Draws the shape of a circle in a zero matrix with the value of alpha
    :param x: 2, array, center
    :param r: integer, radius
    :param alpha: double, opacity
    :return: H,W array, shape matrix
    """
    matrix = np.zeros((H, W), np.dtype('double'))
    cv2.circle(matrix, (x[0], x[1]), r, alpha, thickness=-1)
    return matrix


def draw_image_1(vars):
    """
    Renders N triangles with lineal combination of colors
    :param vars: N, 10 array. (x1,y1,x2,y2,x3,y3,r,g,b,a)
    r,g,b: doubles
    x_i,y_i: integers (will be rounded)
    :return: H,W,3 matrix
    """
    # BACKGROUND_ALPHA = 100
    # BACKGROUND_COLOR = [1., 1., 1.]
    shape_matrix = np.zeros((H, W, N), np.dtype('double'))
    for i in range(N):
        vertex = np.round(np.array(vars[i][0:6])).astype(int)
        # vertex = vertex * (vertex >= 0)  # Coords must be positive
        args = vertex, np.max(vars[i][-1], 0)
        shape_matrix[:, :, i] = triag_matrix(args[0], args[1])
    # shape_matrix[:, :, 0] = np.ones((H, W)) * BACKGROUND_ALPHA
    alpha_sum = np.sum(shape_matrix, axis=2)  # H,W dimension.
    alpha_sum += 1
    RGB = vars[:, 6:9]  # colors
    RGB = RGB * (RGB >= 0)  # Colors must be positive
    out = shape_matrix @ RGB  # the matrix multiplication magic
    for i in range(3):  # Normalization
        out[:, :, i] = (out[:, :, i] / alpha_sum)
    return out


def MAE(imageA, imageB):  # Error absoluto medio (Se asume que ambas imágenes tienen el mismo tamaño)
    err = np.sum(np.abs(imageA.astype("float") - imageB.astype("float")))
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def MSE(imageA, imageB): #Error cuadrático medio (Se asume que ambas imágenes tienen el mismo tamaño)
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def SSIM(i1, i2):
    return 1-ssim(i1, i2, data_range=(np.max(i2) - np.min(i2)), multichannel=True)

# def SSIM_custom(i1, i2):
#     cuad_1 = i1[:int(N/2)]
#     cuad_2 =
#     cuad_3 =
#     cuad_4 =
#     err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
#     err /= float(imageA.shape[0] * imageA.shape[1])
#     return 0


def get_baricenter_from_image(img):
    c1 = img[:, :, 0]
    c2 = img[:, :, 1]
    c3 = img[:, :, 2]
    if np.sum(c1)==0:
        v1 = (100,100)
    else:
        v1 = center_of_mass(c1)
    if np.sum(c1)==0:
        v2 = (100,100)
    else:
        v2 = center_of_mass(c2)
    if np.sum(c1)==0:
        v3 = (100,100)
    else:
        v3 = center_of_mass(c3)
    out = np.stack((v1, v2, v3), axis=0)
    return out

def COM(i1,i2):
    bar1 = get_baricenter_from_image(i1)
    bar2 = get_baricenter_from_image(i2)
    return np.sum(np.multiply(bar1-bar2,bar1-bar2))


def Combined(i1,i2):
    return MSE(i1,i2) + 0.000001*COM(i1,i2)


def partial_dif(i, _vars, _norm, _IMO, _N):
    """
    Numerical partial derivate
    """
    if i % 10 in [6, 7, 8, 9]:
        delta = 0.1
        return 0
    else:
        delta = 10
    current_im = draw_image_1(_vars)
    current_distance = _norm(_IMO, current_im)
    next_vars = np.copy(np.ravel(_vars))
    next_vars[i] += delta
    next_vars.shape = _N, 10
    next_image = draw_image_1(next_vars)
    next_distance = _norm(_IMO, next_image)
    return (next_distance - current_distance) / delta


def numerical_grad(vars, norm, IMO):
    """
    Returns an array with the gradient computed with the passed norm
    """
    # with mp.Pool() as p:
    #     grad = p.starmap(partial_dif, [(i, vars, norm, IMO, N) for i in range(len(np.ravel(vars)))])
    grad = np.asarray([partial_dif(i, vars, norm, IMO, N) for i in range(len(np.ravel(vars)))])
    grad.shape = N, 10
    return grad


def get_random_start():
    """
    Returns a random set of coordinates for the
    :param seed: integer
    :return: vars
    """
    vars = np.zeros((N, 10))
    for i in range(N):
        vars[i] = np.array([np.random.randint(0, W), np.random.randint(0, H),
                            np.random.randint(0, W), np.random.randint(0, H),
                            np.random.randint(0, W), np.random.randint(0, H),
                            1, 1,
                            1, 100])
    # for i in range(N):
    #     vars[i] = np.array([np.random.randint(int(3*W/5), int(4*W/5)), np.random.randint(int(3*H/5), int(4*H/5)),
    #                         np.random.randint(int(3*W/5), int(4*W/5)), np.random.randint(int(3*H/5), int(4*H/5)),
    #                         np.random.randint(int(3*W/5), int(4*W/5)), np.random.randint(int(3*H/5), int(4*H/5)),
    #                         1, 1,
    #                         1, 100])
        #
        # 1, 1,
        # 1, 100])
        #                     np.random.uniform(0, 1), np.random.uniform(0, 1),
        #                     np.random.uniform(0, 1), np.random.uniform(0, 1)])
    return vars





def simple_gradient(target_image, norm, step, max_iter, tol):
    """
    Simple gradient descent.
    :param target_image:
    :param norm:
    :param step: function, takes a integer.
    :param max_iter: just in case.
    :return:
    """

    x_i = get_random_start()
    it = 0
    while it < max_iter:
        print('computing gradient...')
        tt = time.time()
        grad = numerical_grad(x_i, norm, target_image)
        print("Iteration: {0}, Elapsed time: {1}".format(it, time.time() - tt))
        x_next = x_i - step(it) * grad
        difference = norm(draw_image_1(x_i), target_image)
        print("Gradient: {0}".format(grad))
        print("Difference from target: {0}".format(difference))
        if difference < tol:
            x_i = x_next
            break
        x_i = x_next
        image = draw_image_1(x_i)
        # plt.imsave("It {0}.png".format(it), image)
        cv2.namedWindow("Avance")
        cv2.moveWindow("Avance", 780, 250)

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        bottomLeftCornerOfText1 = (5, 365)
        bottomLeftCornerOfText2 = (5, 345)
        bottomLeftCornerOfText3 = (5, 325)
        bottomLeftCornerOfText4 = (5, 305)
        fontScale = 0.5
        fontColor = (255, 255, 255)
        lineType = 1

        # cv2.putText(image, 'Iteration {0}'.format(it),
        #             bottomLeftCornerOfText4,
        #             font,
        #             fontScale,
        #             fontColor,
        #             lineType)
        # cv2.putText(image, 'Method:= SG || N:= {0} || Mesure:= MSE'.format(N),
        #             bottomLeftCornerOfText1,
        #             font,
        #             fontScale,
        #             fontColor,
        #             lineType)
        # cv2.putText(image, 'Elapsed time:={0} s'.format(np.around(time.time() - tt,6)),
        #             bottomLeftCornerOfText2,
        #             font,
        #             fontScale,
        #             fontColor,
        #             lineType)
        # cv2.putText(image,
        #             'Objective Value:= {0}'.format(np.around(difference, 6)),
        #             bottomLeftCornerOfText3,
        #             font,
        #             fontScale,
        #             fontColor,
        #             lineType)



        cv2.imshow("Avance", image)
        cv2.waitKey(1)
        it += 1
    return x_i


def accelerated_descent(target_image, norm, max_iter, tol):
    """

    :param target_image:
    :param norm:
    :param max_iter:
    :param tol:
    :return:
    """
    L = 1
    theta = lambda n: 2 / (n + 2)
    x_i = get_random_start()
    z_i = np.copy(x_i)
    it = 0
    while it < max_iter:
        if it % 10 == 0: print(it)
        y_i = (1 - theta(it)) * x_i + theta(it) * z_i
        grad = numerical_grad(y_i, norm, target_image)
        z_next = z_i - grad / (theta(it) * L)
        x_next = (1 - theta(it)) * x_i + theta(it) * z_next
        if norm(draw_image_1(x_i), draw_image_1(x_next)) < tol:
            x_i = x_next
            break
        x_i = x_next
        z_i = z_next
        it += 1
    return x_i


def greedy_descent(target_image, norm, step, max_iter, tol):
    """
    greedy coordinate descent.
    :param target_image:
    :param norm:
    :param step: function, takes a integer.
    :param max_iter: just in case.
    :return:
    """

    x_i = get_random_start()
    it = 0
    while it < max_iter:
        print('computing gradient...')
        tt = time.time()
        grad = numerical_grad(x_i, norm, target_image)
        print("Iteration: {0}, Elapsed time: {1}".format(it, time.time() - tt))
        n_ = (grad == np.max(grad))
        x_next = x_i - step(it) * n_
        difference = norm(draw_image_1(x_i), target_image)
        print("Gradient: {0}".format(grad))
        print("Difference from target: {0}".format(difference))
        if difference < tol:
            x_i = x_next
            break
        x_i = x_next
        cv2.imshow("Objective", draw_image_1(x_i))
        cv2.waitKey(1)
        it += 1
    return x_i


# variables in order (x1,y1,x2,y2,x3,y3,r,g,b,a)
vars = np.zeros((N, 10))
# vars[0] = np.array([100, 100, 200, 100, 100, 200, .6, .3, .8, .9])
# vars[1] = np.array([150, 100, 250, 100, 50, 200, 0, 1, .3, .1])

for i in range(N):
    vars[i] = np.array([np.random.randint(0, W), np.random.randint(0, H),
                        np.random.randint(0, W), np.random.randint(0, H),
                        np.random.randint(0, W), np.random.randint(0, H),
                        np.random.uniform(0, 1), np.random.uniform(0, 1),
                        np.random.uniform(0, 1), np.random.uniform(0, 1)])

# img_objective = cv2.imread("simple_objective.png")
img_objective = cv2.imread("Little star.png")
# img_objective = cv2.imread("2_rectangles.png")
# img_objective = cv2.imread("Circle.png")
# img_objective = cv2.imread("Star.png")
# img_objective = cv2.imread("Arkadi.png")
img_objective = img_objective / 255
img_test = draw_image_1(vars)

v = get_random_start()

if __name__ == "__main__":
    cv2.namedWindow("Objetivo Simple")
    cv2.moveWindow("Objetivo Simple", 500, 250);
    cv2.imshow("Objetivo Simple", img_objective)
    cv2.waitKey(1)
    # x = simple_gradient(img_objective, Combined, lambda n: 1000, 10000, 1e-9)
    # x = simple_gradient(img_objective, SSIM, lambda n: 1000, 10000, 1e-9)
    x = simple_gradient(img_objective, MSE, lambda n: 1, 10000, 1e-9)

    # img_1 = np.full((400, 400, 3), 1, np.uint8)
    # cv2.circle(img_1, (200, 200), 40, (0, 0, 255), -1)
    #
    # img_2 = np.full((400, 400, 3), 1, np.uint8)
    # cv2.circle(img_2, (200, 201), 40, (0, 0, 255), -1)
    #
    # cv2.namedWindow("IM1")
    # cv2.moveWindow("IM1", 300, 250);
    # cv2.imshow("IM1", img_1)
    # cv2.waitKey(0)
    #
    # cv2.namedWindow("IM2")
    # cv2.moveWindow("IM2", 850, 250);
    # cv2.imshow("IM2", img_2)
    # cv2.waitKey(0)
    #
    # print(get_baricenter_from_image(img_1))
    # print(get_baricenter_from_image(img_2))
    # print(COM(img_1,img_2))
    quit()