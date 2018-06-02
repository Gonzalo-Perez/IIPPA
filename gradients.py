import numpy as np

import multiprocessing as mp

from measures import *







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

