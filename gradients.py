import numpy as np
from measures import *
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


def partial_dif_2_1(i, _vars, norm_mode, _IMO, _delta=.2, scheme=0):
    """
    Numerical Partial derivative, build for parallelization.
    :param i: integer, coordinate
    :param _vars: N,10 arraythe variables from which to evaluate de function
    :param norm_mode: integer, the norm used to compute de difference
    :param _IMO: H,W array, objective image
    :param _delta: float, step for the differentiation
    :param scheme: int, {0,1,2} 0: simple  1: two points   2: Five-point stencil, other: simple
    :return:
    """
    H, W = _IMO.shape[0], _IMO.shape[1]
    delta = _delta
    if scheme == 1:
        images = draw_multi_image_2(_vars, H, W, i, (delta, -delta))
        d1 = general_norm_1(images[0], _IMO, norm_mode)
        d2 = general_norm_1(images[1], _IMO, norm_mode)
        return (d1 - d2) / (2 * delta)
    elif scheme == 2:
        images = draw_multi_image_2(_vars, H, W, i, (2 * delta, delta, -delta, -2 * delta))
        d1 = general_norm_1(images[0], _IMO, norm_mode)
        d2 = general_norm_1(images[1], _IMO, norm_mode)
        d3 = general_norm_1(images[2], _IMO, norm_mode)
        d4 = general_norm_1(images[3], _IMO, norm_mode)
        return ((8 * d2 + d4) - (8 * d3 + d1)) / (12 * delta)
    else:
        images = draw_multi_image_2(_vars, H, W, i, (0, delta))
        d1 = general_norm_1(images[0], _IMO, norm_mode)
        d2 = general_norm_1(images[1], _IMO, norm_mode)
        return (d2 - d1) / delta


def partial_diff_best(vars, H, W, precom_layers, norm_mode, IMO, i, delta, scheme, warm_start):
    """
    Partial derivative that integrates all the functionalities. refer to the num_grad best for information on
    the parameters used.
    :param vars:
    :param H:
    :param W:
    :param norm_mode:
    :param IMO:
    :param i:
    :param delta:
    :param scheme:
    :param warm_start:
    :return:
    """

    if scheme == 1:
        per = (delta, - delta)
        images = draw_multi_image_best(vars, H, W, i, per, crop=warm_start, precomp_layers=precom_layers)
        if warm_start:
            window = get_diff_window_efficiently_triags(vars, i, per, H, W)
            # numpy magic to get index for submatrix
            index = np.ix_(np.arange(window[2], window[3]), np.arange(window[0], window[1]), [0, 1, 2])
            O = IMO[index]
            d1 = general_norm_ws(images[0], O, H, W, norm_mode)
            d2 = general_norm_ws(images[1], O, H, W, norm_mode)
            return (d1 - d2) / (2 * delta)
        else:
            d1 = general_norm_1(images[0], IMO, norm_mode)
            d2 = general_norm_1(images[1], IMO, norm_mode)
            return (d1 - d2) / (2 * delta)
    elif scheme == 2:
        per = (2 * delta, delta, -delta, -2 * delta)
        images = draw_multi_image_best(vars, H, W, i, per, crop=warm_start, precomp_layers=precom_layers)
        if warm_start:
            window = get_diff_window_efficiently_triags(vars, i, per, H, W)
            index = np.ix_(np.arange(window[2], window[3]), np.arange(window[0], window[1]), [0, 1, 2])
            O = IMO[index]
            d1 = general_norm_ws(images[0], O, H, W, norm_mode)
            d2 = general_norm_ws(images[1], O, H, W, norm_mode)
            d3 = general_norm_ws(images[2], O, H, W, norm_mode)
            d4 = general_norm_ws(images[3], O, H, W, norm_mode)
            return ((8 * d2 + d4) - (8 * d3 + d1)) / (12 * delta)
        else:
            d1 = general_norm_1(images[0], IMO, norm_mode)
            d2 = general_norm_1(images[1], IMO, norm_mode)
            d3 = general_norm_1(images[2], IMO, norm_mode)
            d4 = general_norm_1(images[3], IMO, norm_mode)
            return ((8 * d2 + d4) - (8 * d3 + d1)) / (12 * delta)
    else:
        per = (delta, 0)
        images = draw_multi_image_best(vars, H, W, i, per, crop=warm_start, precomp_layers=precom_layers)
        if warm_start:
            window = get_diff_window_efficiently_triags(vars, i, per, H, W)
            # numpy magic to get index for submatrix
            index = np.ix_(np.arange(window[2], window[3]), np.arange(window[0], window[1]), [0, 1, 2])
            O = IMO[index]
            d1 = general_norm_ws(images[0], O, H, W, norm_mode)
            d2 = general_norm_ws(images[1], O, H, W, norm_mode)
            return (d2 - d1) / (-delta)
        else:
            d1 = general_norm_1(images[0], IMO, norm_mode)
            d2 = general_norm_1(images[1], IMO, norm_mode)
            return (d2 - d1) / (-delta)


def partial_dif_warm_start(i, _vars, norm_mode, _IMO, _delta=.2, scheme=0):
    """ INTENDED FOR CONTINIUS VARIABLES
    Numerical Partial derivative. Implements the idea of the "warm start". ie it will only calculate the
    difference in the part of the image that is changing.
    :param i: integer, coordinate
    :param _vars: N,10 arraythe variables from which to evaluate de function
    :param norm_mode: integer, the norm used to compute de difference
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
        O = _IMO[index]
        d1 = general_norm_ws(im0, O, H, W, norm_mode)
        d2 = general_norm_ws(im1, O, H, W, norm_mode)
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
        index = np.ix_((0, 1, 2, 3), np.arange(window[0], window[1] + 1), np.arange(window[2], window[3] + 1),
                       [0, 1, 2])
        i_IMO = np.ix_(np.arange(window[0], window[1] + 1), np.arange(window[2], window[3] + 1), [0, 1, 2])
        ims = images[index]
        O = _IMO[i_IMO]
        d1 = general_norm_ws(ims[0], O, H, W, norm_mode)
        d2 = general_norm_ws(ims[1], O, H, W, norm_mode)
        d3 = general_norm_ws(ims[2], O, H, W, norm_mode)
        d4 = general_norm_ws(ims[3], O, H, W, norm_mode)
        return ((8 * d2 + d4) - (8 * d3 + d1)) / (12 * delta)
    else:
        images = draw_multi_image_2(_vars, H, W, i, (0, delta))
        window = get_diff_window(images[0], images[1])
        # numpy magic to get index for submatrix
        index = np.ix_(np.arange(window[0], window[1] + 1), np.arange(window[2], window[3] + 1), [0, 1, 2])
        im0 = images[0][index]
        im1 = images[1][index]
        O = _IMO[index]
        d1 = general_norm_ws(im0, O, H, W, norm_mode)
        d2 = general_norm_ws(im1, O, H, W, norm_mode)
        return (d2 - d1) / delta
    pass


def get_diff_window(i1, i2):
    """ (mind the reversed names of x,y)
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
    return y_0, y_1, x_0, x_1


def get_diff_window_efficiently_triags(vars, i, deltas, H, W):
    """
    CONSIDERS ONLY STANDARD PARAMETRIZATION OF TRIANGLES.
    Returns the window in which the perturbation of the variable matters without drawing the image.
    Note that the windows computed with this method and the above one are sometimes different.
    This one is always a little bigger.
    :param vars: set of variables for triangles
    :param i: index of variable to perturbate
    :param deltas: array of perturbations for vars[i]
    :param H: Height
    :param W: width
    :return: x_min, x_max, y_min, y_max
    """
    i_triag = i // 10
    v = vars[i_triag]  # afected triangle variables.
    if i % 10 < 6:
        j = i % 10
        if j % 2 == 0:
            M = len(deltas)
            x1 = np.ones(M, dtype=float) * v[0] + (j == 0) * np.array(deltas)
            x2 = np.ones(M, dtype=float) * v[2] + (j == 2) * np.array(deltas)
            x3 = np.ones(M, dtype=float) * v[4] + (j == 4) * np.array(deltas)
            y1, y2, y3 = v[1], v[3], v[5]
        else:
            M = len(deltas)
            x1, x2, x3 = v[0], v[2], v[4]
            y1 = np.ones(M, dtype=float) * v[1] + (j == 1) * np.array(deltas)
            y2 = np.ones(M, dtype=float) * v[3] + (j == 3) * np.array(deltas)
            y3 = np.ones(M, dtype=float) * v[5] + (j == 5) * np.array(deltas)
        x_min = np.min(np.array((x1, x2, x3)) * W).astype(int)
        x_max = np.max(np.array((x1, x2, x3)) * W).astype(int)
        y_min = np.min(np.array((y1, y2, y3)) * H).astype(int)
        y_max = np.max(np.array((y1, y2, y3)) * H).astype(int)
        pass
    else:
        # Color change.
        x_min = np.min((int(v[0] * W), int(v[2] * W), int(v[4] * W)))
        x_max = np.max((int(v[0] * W), int(v[2] * W), int(v[4] * W)))
        y_min = np.min((int(v[1] * H), int(v[3] * H), int(v[5] * H)))
        y_max = np.max((int(v[1] * H), int(v[3] * H), int(v[5] * H)))
        pass
    x_min -= 1
    y_min -= 1
    x_max += 1
    y_max += 1
    x_min = np.max((0, x_min))
    x_max = np.min((W, x_max))
    y_min = np.max((0, y_min))
    y_max = np.min((H, y_max))
    return x_min, x_max, y_min, y_max


def numerical_grad(vars, norm, IMO, delta=.2, _scheme=0):
    """
    Returns the gradient computed numerically, considering the passed norm
    :param vars: vars from which to evaluate
    :param norm: norm to consider the difference
    :param IMO: Objective image
    :param delta: step of the numerical derivative.
    :param _scheme: argument to pass to the partial differentiation method (integer)
    :return: array, gradient
    """
    N = len(vars)
    grad = np.asarray([partial_dif_2
                       (i, vars, norm, IMO, _delta=delta, scheme=_scheme) for i in range(len(np.ravel(vars)))])
    grad = np.asarray(grad)
    grad.shape = N, 10
    return grad


def numerical_grad_2(vars, norm_mode, IMO, delta=.2, _scheme=0, parallel=True):
    """ Fixed for parallelization
    Returns the gradient computed numerically, considering the passed norm
    :param vars: vars from which to evaluate
    :param norm_mode: integer, norm to consider de difference
    :param IMO: Objective image
    :param delta: step for the numerical differentiation
    :param _scheme: argument to pass to the partial differentiation method (integer)
    :param parallel: boolean, use parallelization
    :return: array, gradient
    """
    N = len(vars)
    if parallel:
        with mp.Pool() as p:
            grad = p.starmap(partial_dif_2_1,
                             [(i, vars, norm_mode, IMO, delta, _scheme) for i in range(len(np.ravel(vars)))])
    else:
        grad = np.asarray([partial_dif_2_1
                           (i, vars, norm_mode, IMO, _delta=delta, scheme=_scheme) for i in range(len(np.ravel(vars)))])
    grad = np.asarray(grad)
    grad.shape = N, 10
    return grad


def num_stochastic_grad(vars, norm_mode, IMO, ratio_computed=.2, hard_max=-1, delta=.2, _scheme=0, parallel=True,
                        choose_triags=False):
    """
    Computes a simple numeric stochastic gradient. Each call picks randomly Num_vars*ratio_computed vars to compute.
    The others are set to zero.
    :param vars: vars value to compute gradient
    :param norm_mode: norm arg to be passed
    :param IMO: objective image.
    :param ratio_computed: percentage of total variables to compute.
    :param hard_max: max num of directions to compute.
    :param delta: step for the numerical differentiation
    :param _scheme: integer, scheme type for the numerical differentiation
    :param parallel: bool, true for parallelization
    :param choose_triags: bool, if true whole triangles are choosen instead of individual variables.
    :return:
    """
    num_vars = len(np.ravel(vars))
    if choose_triags:
        num_triags = len(vars)
        if hard_max > 0:
            size = int(np.min((np.ceil(num_triags * ratio_computed), hard_max)))
        else:
            size = int(np.ceil(num_triags * ratio_computed))
        triag_choice = np.sort(np.random.choice(np.arange(num_triags), size, False))
        choice = np.array([10 * i + j for i in triag_choice for j in range(10)])
    else:
        if hard_max > 0:
            size = np.min((round(num_vars * ratio_computed), hard_max))
        else:
            size = round(num_vars * ratio_computed)
        choice = np.sort(np.random.choice(np.arange(num_vars), size, False))
    N = len(vars)
    grad = np.zeros(num_vars)
    if parallel:
        with mp.Pool() as p:
            grad[choice] = p.starmap(partial_dif_2_1,
                                     [(i, vars, norm_mode, IMO, delta, _scheme) for i in choice])
    else:
        grad[choice] = np.asarray([partial_dif_2_1
                                   (i, vars, norm_mode, IMO, _delta=delta, scheme=_scheme) for i in choice])
    grad = np.asarray(grad)
    grad.shape = N, 10
    return grad


def num_grad_best(vars, norm_mode, IMO, index, delta=.2, scheme=0, warm_start=True, parallel=True):
    """
    Computes numerical gradient. Integrates all functionality, different differentiation schemes, the warm start,
    parallelization, optimization for redundant calculations, variations in the norm and the posibility to
    choose which variables to ignore.
    :param vars: variables, N,10 matrix
    :param norm_mode: integer, to chose norm mode
    :param IMO: H,W,3 array, the objective image
    :param index: array, subset of the index of vars.
    :param delta: float, step for the numerical differentiation
    :param scheme: integer, scheme for the numerical differentiation
    :param warm_start: bool, if true the warm start optimization is used
    :param parallel: bool, if true multiple cores are used in parallel to compute the different partial derivatives.
    :return: 10*N array,
    """
    N = len(vars)
    H, W = IMO.shape[0], IMO.shape[1]
    grad = np.zeros((10 * N), dtype=float)
    precom_layers = precompute_shape_layers(vars, H, W, use_treads=parallel)
    if parallel:
        with mp.Pool() as p:
            computed = p.starmap(partial_diff_best,
                                 [(vars, H, W, precom_layers, norm_mode, IMO, i, delta, scheme, warm_start)
                                  for i in index])
        grad[index] = computed
    else:
        for i, c in enumerate(index):
            grad[c] = partial_diff_best(vars, H, W, precom_layers, norm_mode, IMO, i, delta, scheme, warm_start)
    return grad.reshape(N, 10)


def grad_warm_start(vars, norm, IMO, delta=.2, scheme=0, parallel=False):
    """ This needs atention to work well. other norms can be modified to be compatible.
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
            grad = p.starmap(partial_dif_warm_start,
                             [(i, vars, norm, IMO, delta, scheme) for i in range(len(np.ravel(vars)))])
    else:
        grad = np.asarray([partial_dif_warm_start
                           (i, vars, norm, IMO, _delta=delta, scheme=scheme) for i in range(len(np.ravel(vars)))])
    grad = np.asarray(grad)
    grad.shape = N, 10
    return grad
