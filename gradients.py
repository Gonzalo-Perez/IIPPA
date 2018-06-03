import numpy as np
import multiprocessing as mp
from draw import *


def partial_dif_2(i, _vars, _norm, _IMO, _delta=.2, scheme=0):
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
    H, W = _IMO.shape[0], _IMO.shape[1]
    delta = _delta
    if scheme == 1:
        images = draw_multi_image_2(_vars, H, W, i, (delta, -delta))
        d1 = _norm(images[0], _IMO)
        d2 = _norm(images[1], _IMO)
        return (d1 - d2) / (2 * delta)
    elif scheme == 2:
        images = draw_multi_image_2(_vars, H, W, i, (2 * delta, delta, -delta, -2 * delta))
        d1 = _norm(images[0], _IMO)
        d2 = _norm(images[1], _IMO)
        d3 = _norm(images[2], _IMO)
        d4 = _norm(images[3], _IMO)
        return ((8 * d2 + d4) - (8 * d3 + d1)) / (12 * delta)
    else:
        images = draw_multi_image_2(_vars, H, W, i, (0, delta))
        d1 = _norm(images[0], _IMO)
        d2 = _norm(images[1], _IMO)
        return (d2 - d1) / delta


def partial_dif_3(i, _vars, _norm, _IMO, _delta=.2, scheme=0):
    """ INTENDED FOR CONTINIUS VARIABLES
        Numerical Partial derivative. Implements the idea of the "warm start". ie it will only calculate the
        difference in the part of the image that is changing.
        :param i: integer, coordinate
        :param _vars: N,10 arraythe variables from which to evaluate de function
        :param _norm: function, the norm used to compute de difference
        :param _IMO: H,W array, objective image
        :param _N: integer, global parameter, number of shapes used
        :param _delta: float, step for the differentiation
        :param scheme: int, {0,1,2} 0: simple  1: two points   2: Five-point stencil, other: simple
        :return:
        """
    H, W = _IMO.shape[0], _IMO.shape[1]
    delta = _delta
    if scheme == 1:
        images = draw_multi_image_2(_vars, H, W, i, (delta, -delta))
        window = get_diff_window(images[0], images[1])
        # numpy magic to get index for submatrix
        index = np.ix_(np.arange(window[0], window[1] + 1), np.arange(window[2], window[3] + 1), [0, 1, 2])
        im0 = images[0][index]
        im1 = images[1][index]
        d1 = _norm(im0, _IMO)
        d2 = _norm(im1, _IMO)
        return (d1 - d2) / (2 * delta)
    elif scheme == 2:
        images = draw_multi_image_2(_vars, H, W, i, (2 * delta, delta, -delta, -2 * delta))
        w1 = get_diff_window(images[0], images[1])
        w2 = get_diff_window(images[0], images[2])
        w3 = get_diff_window(images[0], images[3])
        # This comparision is transitive, so this is enough
        window = np.min((w1[0], w2[0], w3[0])), np.max((w1[1], w2[1], w3[1])), \
                 np.min((w1[2], w2[2], w3[2])), np.max((w1[3], w2[3], w3[3]))
        # numpy magic to get index for submatrix
        index = np.ix_(np.arange(window[0], window[1] + 1), np.arange(window[2], window[3] + 1), [0, 1, 2])
        d1 = _norm(images[0][index], _IMO)
        d2 = _norm(images[1][index], _IMO)
        d3 = _norm(images[2][index], _IMO)
        d4 = _norm(images[3][index], _IMO)
        return ((8 * d2 + d4) - (8 * d3 + d1)) / (12 * delta)
    else:
        images = draw_multi_image_2(_vars, H, W, i, (0, delta))
        window = get_diff_window(images[0], images[1])
        # numpy magic to get index for submatrix
        index = np.ix_(np.arange(window[0], window[1] + 1), np.arange(window[2], window[3] + 1), [0, 1, 2])
        i1 = images[0][index]
        i2 = images[1][index]
        d1 = _norm(i1, _IMO[index])
        d2 = _norm(i2, _IMO[index])
        return (d2 - d1) / delta
    pass


def get_diff_window(i1, i2):
    """
    Returns the start and end of the window where to images are different.
    Expects H,W,3 images.
    :param i1:
    :param i2:
    :return:
    """
    diff = i1 != i2
    index = np.nonzero(diff)
    x, y = index[0], index[1]
    x_0 = np.min(x)
    x_1 = np.max(x)
    y_0 = np.min(y)
    y_1 = np.max(y)
    return x_0, x_1, y_0, y_1


def numerical_grad(vars, norm, IMO, delta=.2, _scheme=0, parallel=False):
    """
    Returns the gradient computed numerically, considering the passed norm
    :param vars: vars from which to evaluate
    :param norm: norm to consider the difference
    :param IMO: Objective image
    :param N: number of triangles
    :param scheme: argument to pass to the partial differentiation method (integer)
    :param parallel: boolean, use parallelization
    :return: array, gradient
    """
    N = len(vars)
    if parallel:
        with mp.Pool() as p:
            grad = p.starmap(partial_dif_2,
                             [(i, vars, norm, IMO, delta, _scheme) for i in range(len(np.ravel(vars)))])
    else:
        grad = np.asarray([partial_dif_2
                           (i, vars, norm, IMO, _delta=delta, scheme=_scheme) for i in range(len(np.ravel(vars)))])
    grad = np.asarray(grad)
    grad.shape = N, 10
    return grad


def grad_warm_start(vars, norm, IMO, delta=.2, scheme=0, parallel=False):
    """
    Returns the gradient computed numerically. The norm MUST be a SE-like. ie: must be a sum of a function(p1_i,p2_i),
    that being that only considers pixels locally. even the normalization of MSE breaks the optimization.
    :param vars:
    :param IMO:
    :param delta:
    :param scheme:
    :param parallel:
    :return:
    """
    N = len(vars)
    if parallel:
        with mp.Pool() as p:
            grad = p.starmap(partial_dif_3,
                             [(i, vars, norm, IMO, delta, scheme) for i in range(len(np.ravel(vars)))])
    else:
        grad = np.asarray([partial_dif_3
                           (i, vars, norm, IMO, _delta=delta, scheme=scheme) for i in range(len(np.ravel(vars)))])
    grad = np.asarray(grad)
    grad.shape = N, 10
    return grad
