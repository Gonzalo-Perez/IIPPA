import numpy as np
import cv2
import random
from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_mse as MSE
import multiprocessing as mp
import time
from scipy.ndimage.measurements import center_of_mass
import numexpr as ne

# np.random.seed(1)
# random.seed(1)
H = 100
W = 100
N = 15
MM = np.array([[W, H],  # DO NOT DELETE!
               [W, H],
               [W, H]], np.dtype('float'))

T1 = 0
T2 = 0


def triag_matrix_1(x, alpha):
    """ USES INTEGERS FOR POSITIONS
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


# LAST VERSION
def triag_matrix_2(x, alpha):
    """ USED DOUBLES FOR POSITIONS
    Draws the shape of a triangle in a zero matrix with the value of alpha
    :param x: 3,2 array, the coords of the vertex
    :param alpha: double, opacity
    :return: H,W array, shape matrix
    """
    matrix = np.zeros((H, W), np.dtype('float'))
    if x.shape != (3, 2):
        x.shape = 3, 2
    x = x * MM
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


def variable_transformation(v):
    """ v = (x, y, phi_1, phi_2, rho_1, rho_2)
    phi_1: angle from horizontal to first point
    phi_2: angle from first point to second point
    rho_i: distance from fixed point to point_i

    Transform v from mixed polar to cartesian coordinates
    :param v: v = (x, y, phi_1, phi_2, rho_1, rho_2, R, G, B, alpha)
    :return: x = (x1, y1, x2, y2, x3, y3, R, G, B, alpha)
    """
    out = np.zeros_like(v)
    out[6:10] = v[6:10]
    out[0:2] = v[0:2]
    out[2] = v[4] * np.cos(2 * np.pi * v[2])
    out[3] = v[4] * np.sin(2 * np.pi * v[2])
    out[4] = v[5] * np.cos(2 * np.pi * (v[2] + v[3]))
    out[5] = v[5] * np.sin(2 * np.pi * (v[2] + v[3]))
    return out


def draw_image_1(vars):
    """ USES INTEGERS FOR POSITIONS
    Renders N triangles with lineal combination of colors
    :param vars: N, 10 array. (x1,y1,x2,y2,x3,y3,r,g,b,a)
    r,g,b: doubles
    x_i,y_i: integers (will be rounded)
    :return: H,W,3 matrix
    """
    shape_matrix = np.zeros((H, W, N), np.dtype('double'))
    for i in range(N):
        vertex = np.round(np.array(vars[i][0:6])).astype(int)
        # vertex = vertex * (vertex >= 0)  # Coords must be positive
        args = vertex, np.max(vars[i][-1], 0)
        shape_matrix[:, :, i] = triag_matrix_1(args[0], args[1])
    alpha_sum = np.sum(shape_matrix, axis=2)  # H,W dimension.
    alpha_sum += 1
    RGB = vars[:, 6:9]  # colors
    RGB = RGB * (RGB >= 0)  # Colors must be positive
    out = shape_matrix @ RGB  # the matrix multiplication magic
    for i in range(3):  # Normalization
        out[:, :, i] = (out[:, :, i] / alpha_sum)
    return out


# LAST VERSION
def draw_image_2(vars):
    """ USED DOUBLES FOR POSITIONS
    FIXED, base dimming
    Renders N triangles with lineal combination of colors
    :param vars: N, 10 array. (x1,y1,x2,y2,x3,y3,r,g,b,a)
    r,g,b: doubles
    x_i,y_i: integers (will be rounded)
    :return: H,W,3 matrix
    """
    global T1, T2
    tt = time.time()

    BACKGROUND_ALPHA = .2
    BACKGROUND_COLOR = [1., 1., 1.]
    shape_matrix = np.zeros((H, W, N + 1), np.dtype('float'))
    for i in range(N):
        vertex = np.array(vars[i][0:6])
        args = vertex, np.max(vars[i][-1], 0)
        shape_matrix[:, :, i + 1] = triag_matrix_2(args[0], args[1])
    shape_matrix[:, :, 0] = np.ones((H, W)) * BACKGROUND_ALPHA
    T1 += time.time() - tt
    tt = time.time()
    alpha_sum = np.sum(shape_matrix, axis=2)  # H,W dimension.
    RGB = list(vars[:, 6:9])  # colors
    RGB.insert(0, BACKGROUND_COLOR)
    RGB = np.asarray(RGB)
    RGB = RGB * (RGB >= 0)  # Colors must be positive
    out = shape_matrix @ RGB  # the matrix multiplication magic
    for i in range(3):  # Normalization
        out[:, :, i] = (out[:, :, i] / alpha_sum)
    T2 += time.time() - tt
    return out


def draw_multi_image(vars, index, perturbations, _N):
    """
    Renders N triangles with lineal combination of colors.
    Returns k images, where k is the lenght of 'perturbarions'
    Each image has a perturbation from the perturbation array in the vars[index] coordinate.
    :param vars: N,10 array, initial variables.
    :param index: integer, position to make the perturbation
    :param perturbations: 1D array, values of the perturbations (to be added)
    :return: k,H,W array, k images
    """

    K = len(perturbations)
    original_shape_matrix = np.zeros((H, W, N), np.dtype('float'))
    which_layer = index // 10
    v = np.ravel(vars)
    mod_vars = []
    result = []
    for k in range(K):
        v1 = np.copy(v)
        v1[index] += perturbations[k]
        v1.shape = _N, 10
        mod_vars.append(v1)
    mod_vars = np.asarray(mod_vars)
    for i in range(N):
        vertex = np.array(vars[i][0:6])
        args = vertex, np.max(vars[i][-1], 0)
        original_shape_matrix[:, :, i] = triag_matrix_2(args[0], args[1])
    for k in range(K):
        shape_matrix = np.copy(original_shape_matrix)
        i = which_layer
        vertex = np.array(mod_vars[k][i][0:6])
        args = vertex, np.max(mod_vars[k][i][-1], 0)
        shape_matrix[:, :, i] = triag_matrix_2(args[0], args[1])
        alpha_sum = np.sum(shape_matrix, axis=2)  # H,W dimension.
        alpha_sum += (alpha_sum == 0)
        RGB = mod_vars[k][:, 6:9]  # colors
        RGB = RGB * (RGB >= 0)  # Colors must be positive
        out = shape_matrix @ RGB  # the matrix multiplication magic
        for i in range(3):  # Normalization
            out[:, :, i] = (out[:, :, i] / alpha_sum)
        result.append(out)
    return np.stack(result, axis=0)


# LAST VERSION
def draw_multi_image_2(vars, index, perturbations, _N):
    """ (SE PUEDE OPTIMIZAR MAS SI SE DISTINGUE QUE TIPO DE VARIABLE ESTA SIENDO PERTURBADA)
    Renders N triangles with lineal combination of colors.
    Returns k images, where k is the length of 'perturbarions'
    Each image has a perturbation from the perturbation array in the vars[index] (flat) coordinate.
    :param vars: N,10 array, initial variables.
    :param index: integer, position to make the perturbation
    :param perturbations: 1D array, values of the perturbations (to be added)
    :return: k,H,W,3 array, k images
    """
    BACKGROUND_ALPHA = .2
    BACKGROUND_COLOR = [1., 1., 1.]

    K = len(perturbations)
    which_layer = index // 10
    ind = index % 10
    v = list(vars)
    v_layer = v.pop(which_layer)

    shape_matrix = np.zeros((H, W, N), np.dtype('float'))
    for i in range(N - 1):
        vertex = np.array(v[i][0:6])
        args = vertex, np.max(v[i][-1], 0)
        shape_matrix[:, :, i + 1] = triag_matrix_2(args[0], args[1])

    shape_matrix[:, :, 0] = np.ones((H, W), np.dtype('float')) * BACKGROUND_ALPHA  # adds white background
    alpha_sum = np.sum(shape_matrix, axis=2)  # H,W dimension.

    RGB = list(np.asarray(v)[:, 6:9])  # colors N - 1
    RGB.insert(0, BACKGROUND_COLOR)  # insert manually the background
    RGB = np.asarray(RGB)
    RGB = RGB * (RGB >= 0)  # Colors must be positive
    out = shape_matrix @ RGB  # the matrix multiplication magic

    outs = np.zeros((K, H, W, 3))
    for k in range(K):
        u = np.copy(v_layer)
        u[ind] += perturbations[k]
        vertex = np.asarray(u)[0:6]
        args = vertex, np.max(u[-1], 0)
        lay = triag_matrix_2(args[0], args[1])
        a_sum = alpha_sum + lay
        lay.shape = H, W, 1
        col = np.asarray(u[6:9])
        col = col * (col >= 0)
        col.shape = 1, 3
        outs[k] = out + lay @ col
        for i in range(3):  # Normalization
            outs[k][:, :, i] = (out[:, :, i] / a_sum)
    return outs


def MAE(imageA, imageB):  # Error absoluto medio (Se asume que ambas imágenes tienen el mismo tamaño)
    err = np.sum(np.abs(imageA.astype("float") - imageB.astype("float")))
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def MSE(imageA, imageB):  # Error cuadrático medio (Se asume que ambas imágenes tienen el mismo tamaño)
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def SSIM(i1, i2):
    return 1 - ssim(i1, i2, data_range=(np.max(i2) - np.min(i2)), multichannel=True)


def MSE_w_Centroid(i1, i2):
    """
    Returns a convex linear combination of MSE and centroid distance.
    (the centroid distance is normalized)
    :param i1: image 1
    :param i2: image 2
    :param beta: float between 0 and 1. the combination parameter. 0: full centroid. 1: full MSE
    :return: float
    """
    beta = .5
    mse = MSE(i1, i2)
    c1 = get_baricenter_from_image(i1)
    c2 = get_baricenter_from_image(i2)
    n_d = np.linalg.norm(((c1 - c2) * np.array((1. / W, 1. / H))))
    return beta * mse + (1 - beta) * n_d


def MSE_centroid_colormeans(i1, i2):
    """
    Returns a convex linear combination of MSE, centroid and color mean distances.
    :param i1: image 1
    :param i2: image 2
    :return: float
    """
    a, b, c = 1.5, .5, 1
    measures = np.array([MSE(i1, i2),
                         get_centroid_dist_norm(i1, i2),
                         get_channel_means_diff_norm(i1, i2)])
    ponderators = np.array([a, b, c])
    return (measures @ ponderators) / np.sum(ponderators)


def get_centroid_dist_norm(i1, i2):
    """
    Returns the norm of the vector of distance of centroids for each channel
    :param i1: image 1
    :param i2: image 2
    :return: float non negative float
    """
    c1 = get_baricenter_from_image(i1)
    c2 = get_baricenter_from_image(i2)
    n_d = np.linalg.norm(((c1 - c2) * np.array((1. / W, 1. / H))))
    return n_d


def get_channel_means_diff_norm(i1, i2):
    """
    Returns the norm of the vector of differences of means for each channel
    :param i1: image 1
    :param i2: image 2
    :return:
    """
    RGB_1 = np.mean(i1, axis=(0, 1))
    RGB_2 = np.mean(i2, axis=(0, 1))
    diff = np.linalg.norm(RGB_1 - RGB_2)
    return diff


# PENDING TO DEBUG
def get_baricenter_from_vars(vars, _N):  # not tested
    """
    Returns the coordinates of the centroid
    :param vars: triangle variables
    :param _N:
    :return: 3,2 array. colors centroid
    """
    print('WARNING: not working as expected. NEED debugging!')
    V_pos = np.asarray(vars[:, 0:6])
    areas = np.zeros(_N)
    areas.shape = _N, 1
    for i in range(_N):
        tri = V_pos[i]
        tri.shape = 3, 2
        ones = np.ones(3)
        ones.shape = 3, 1
        u = np.concatenate((tri, ones), axis=1)
        areas[i] = 0.5 * np.linalg.det(u)  # hope that this is as efficient as it gets
    V_col = vars[:, 6:9]
    V_alpha = vars[:, 9:10]
    V_alpha = V_alpha * areas
    xy = (1 / 3) * np.array([[1, 0],
                             [0, 1],
                             [1, 0],
                             [0, 1],
                             [1, 0],
                             [0, 1]])
    pos = V_pos @ xy  # returns a N,2 matrix with the centroids
    V_alpha.shape = _N, 1
    V_alpha /= np.sum(V_alpha)  # normalized weights
    weights = np.concatenate((V_alpha, V_alpha), axis=1)  # N,2 matrix
    w_pos = (pos * weights)  # N,2 matrix of weighted coordinates
    colors = np.transpose(V_col)  # 3,N matrix of colors
    out = colors @ w_pos
    return out


def get_baricenter_from_image(img):
    """
    Returns the centroid of the image for each channel
    :param img:
    :return:
    """
    c1 = img[:, :, 0]
    c2 = img[:, :, 1]
    c3 = img[:, :, 2]
    v1 = center_of_mass(c1)
    v2 = center_of_mass(c2)
    v3 = center_of_mass(c3)
    out = np.stack((v1, v2, v3), axis=0)
    return out


def partial_dif(i, _vars, _norm, _IMO, _N):
    """
    Numerical Partial derivative
    :param i: integer, coordinate
    :param _vars: N,10 arraythe variables from which to evaluate de function
    :param _norm: function, the norm used to compute de difference
    :param _IMO: H,W array, objective image
    :param _N: integer, global parameter, number of shapes used
    :return:
    """
    if i % 10 in [6, 7, 8, 9]:
        delta = 0.2
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


# LAST VERSION
def partial_dif_2(i, _vars, _norm, _IMO, _N, _delta=.2, scheme='simple'):
    """ INTENDED FOR CONTINIUS VARIABLES
    Numerical Partial derivative
    :param i: integer, coordinate
    :param _vars: N,10 arraythe variables from which to evaluate de function
    :param _norm: function, the norm used to compute de difference
    :param _IMO: H,W array, objective image
    :param _N: integer, global parameter, number of shapes used
    :param _delta: float, step for the differentiation
    :param scheme: int, {0,1,2} 0: simple  1: two points   2: Five-point stencil, other: simple
    :return:
    """
    # if i % 10 >= 6: return 0
    delta = _delta
    if scheme == 1:
        images = draw_multi_image_2(_vars, i, (delta, -delta), _N)
        d1 = _norm(images[0], _IMO)
        d2 = _norm(images[1], _IMO)
        return (d1 - d2) / (2 * delta)
    elif scheme == 2:
        images = draw_multi_image_2(_vars, i, (2 * delta, delta, -delta, -2 * delta), _N)
        d1 = _norm(images[0], _IMO)
        d2 = _norm(images[1], _IMO)
        d3 = _norm(images[2], _IMO)
        d4 = _norm(images[3], _IMO)
        return ((8 * d2 + d4) - (8 * d3 + d1)) / (12 * delta)
    else:
        images = draw_multi_image_2(_vars, i, (0, delta), _N)
        d1 = _norm(images[0], _IMO)
        d2 = _norm(images[1], _IMO)
        return (d2 - d1) / delta


def numerical_grad(vars, norm, IMO, N):
    """
    Returns an array with the gradient computed with the passed norm
    """
    # with mp.Pool() as p:
    #    grad = p.starmap(partial_dif, [(i, vars, norm, IMO, N) for i in range(len(np.ravel(vars)))])
    grad = np.asarray([partial_dif(i, vars, norm, IMO, N) for i in range(len(np.ravel(vars)))])
    grad = np.asarray(grad)
    grad.shape = N, 10
    return grad


# LAST VERSION
def numerical_grad_2(vars, norm, IMO, N, delta=.2, _scheme=0, parallel=False):
    """
    Returns the gradient computed numericaly, considering the passed norm
    :param vars: vars from which to evaluate
    :param norm: norm to consider the difference
    :param IMO: Objective image
    :param N: number of triangles
    :param scheme: argument to pass to the partial differentiation method (integer)
    :param parallel: boolean, use parallelization
    :return: array, gradient
    """
    if parallel:
        with mp.Pool() as p:
            grad = p.starmap(partial_dif_2,
                             [(i, vars, norm, IMO, N, delta, _scheme) for i in range(len(np.ravel(vars)))])
    else:
        grad = np.asarray([partial_dif_2
                           (i, vars, norm, IMO, N, _delta=delta, scheme=_scheme) for i in range(len(np.ravel(vars)))])
    grad = np.asarray(grad)
    grad.shape = N, 10
    return grad


def get_random_start_1():
    """ USED INTEGERS FOR POSITIONS
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
    return vars


# LAST VERSION
def get_random_start_2():
    """ USED DOUBLES FOR POSITIONS
    Returns a random set of coordinates for the
    :param seed: integer
    :return: vars
    """
    black = False
    vars = np.zeros((N, 10))
    for i in range(N):
        if not black:
            vars[i] = np.array([np.random.uniform(), np.random.uniform(),
                                np.random.uniform(), np.random.uniform(),
                                np.random.uniform(), np.random.uniform(),
                                np.random.uniform(),
                                np.random.uniform(),
                                np.random.uniform(),
                                np.random.uniform()])
        else:
            vars[i] = np.array([np.random.uniform(), np.random.uniform(),
                                np.random.uniform(), np.random.uniform(),
                                np.random.uniform(), np.random.uniform(),
                                0, 0, 0, 10])
    return vars


### ALL ITERATIVE METHODS PENDING TO REFACTOR
def simple_gradient(target_image, norm, step, max_iter, tol, _delta=.2, diff_scheme_to_use=0, use_threads=False,
                    show_progress=False):
    """
    Simple gradient descent
    :param target_image: image to adjust. must have float colors
    :param norm: norm to compare images
    :param step: function, step to consider
    :param max_iter: integer, max number of iterations
    :param tol: double, tolerance to stop iterations
    :param _delta: double, for numerical differentiation calculation
    :param diff_scheme_to_use: {0,1,2} to be passed to numerical differentiation
    :param use_threads: boolean
    :param show_progress: boolean
    :return:
    """

    x_i = get_random_start_2()
    it = 0
    while it < max_iter:
        print('computing gradient...')
        tt = time.time()
        grad = numerical_grad_2(x_i, norm, target_image, N, delta=_delta, _scheme=diff_scheme_to_use,
                                parallel=use_threads)
        print("Iteration: {0}, Elapsed time: {1}".format(it, time.time() - tt))
        print("x: {0}".format(x_i))
        x_next = x_i - step(it) * grad
        difference = norm(draw_image_2(x_i), target_image)
        print("Gradient: {0}".format(grad))
        print("Difference from target: {0}".format(difference))
        if difference < tol:
            x_i = x_next
            break
        x_i = x_next
        if show_progress:
            imagen = draw_image_2(x_i)
            cv2.imshow("Objective", imagen)
            if it % 5 == 0:
                cv2.imwrite('pentagon_iteration{}.png'.format(str(it)), np.array((imagen * 255), np.dtype(int)))
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
        grad = numerical_grad(y_i, norm, target_image, N)
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
    """ WE SHOULD IMPLEMENT A LINESEARCH FOR THE CHOOSEN COORDINATE
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
        grad = numerical_grad(x_i, norm, target_image, N)
        print("Iteration: {0}, Elapsed time: {1}".format(it, time.time() - tt))
        n_ = (grad == np.max(grad))
        if np.count_nonzero(n_) >= 2:
            n_ = np.zeros_like(n_)
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


if __name__ == "__main__":
    img_objective = cv2.imread("circulos_coloridos.png")
    img_objective = (img_objective / 255)

    cv2.namedWindow("Objetivo Simple")
    cv2.moveWindow("Objetivo Simple", 500, 250);
    cv2.imshow("Objetivo Simple", img_objective)
    cv2.waitKey(0)

    x = simple_gradient(img_objective, MSE, lambda n: .3 / np.log(np.e + n), 1000, 1e-9, _delta=.05,
                        diff_scheme_to_use=2, use_threads=False, show_progress=True)
    quit()
