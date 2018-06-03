import numpy as np
import cv2


# LAST VERSION
def triag_matrix_2(x, alpha, H, W):
    """ USED DOUBLES FOR POSITIONS
    Draws the shape of a triangle in a zero matrix with the value of alpha
    :param x: 3,2 array, the coords of the vertex
    :param alpha: double, opacity
    :return: H,W array, shape matrix
    """
    MM = np.array([[W, H],  # DO NOT DELETE!
                   [W, H],
                   [W, H]], np.dtype('float'))
    matrix = np.zeros((H, W), np.dtype('float'))
    if x.shape != (3, 2):
        x.shape = 3, 2
    x = x * MM
    x = x.astype(int)
    cv2.fillConvexPoly(matrix, x, alpha)
    return matrix


def circle_matrix(x, r, alpha, H, W):
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


def draw_image_2(vars, H, W, background_color=(1., 1., 1.), background_alpha=.05):
    """ USED DOUBLES FOR POSITIONS
    FIXED, base dimming
    Renders N triangles with lineal combination of colors
    :param vars: N, 10 array. (x1,y1,x2,y2,x3,y3,r,g,b,a)
    r,g,b: doubles
    x_i,y_i: integers (will be rounded)
    :return: H,W,3 matrix
    """
    N = len(vars)
    shape_matrix = np.zeros((H, W, N + 1), np.dtype('float'))
    for i in range(N):
        vertex = np.array(vars[i][0:6])
        args = vertex, np.max(vars[i][-1], 0)
        shape_matrix[:, :, i + 1] = triag_matrix_2(args[0], args[1], H, W)
    shape_matrix[:, :, 0] = np.ones((H, W)) * background_alpha

    alpha_sum = np.sum(shape_matrix, axis=2)  # H,W dimension.
    RGB = list(vars[:, 6:9])  # colors
    RGB.insert(0, background_color)
    RGB = np.asarray(RGB)
    RGB = RGB * (RGB < 1) * (RGB >= 0) + (RGB >= 1) # Colors must be between 0 and 1.
    out = shape_matrix @ RGB  # the matrix multiplication magic
    for i in range(3):  # Normalization
        out[:, :, i] = (out[:, :, i] / alpha_sum)
    return out


def draw_multi_image_2(vars, H, W, index, perturbations, background_color=(1., 1., 1.),
                       background_alpha=.05, return_submatrix_coords=False):
    """ THE SUBMATRIX COORDS ARE NOT WORKING
        (there is more space for optimization if the kind of the perturbated variable is taken into account)
        (there is more space for optimization if the submatrix coords take into account the channels)
    Renders N triangles with lineal combination of colors.
    Returns k images, where k is the length of 'perturbarions'
    Each image has a perturbation from the perturbation array in the vars[index] (flat) coordinate.
    If return_submatrix_coords=True the function also returns the coords of the submatrix in which there is change.
    :param vars: N,10 array, initial variables.
    :param index: integer, position to make the perturbation
    :param perturbations: 1D array, values of the perturbations (to be added)
    :param background_color: (r,g,b), colors between 0 and 1.
    :param background_alpha: positive float. ZERO will cause error
    :param return_submatrix_coords: boolean, if true the submatrix windows coordinates are computed and returned
    :return: k,H,W,3 array, k images (and ((x1,y1),(x2,y2)) conditioned on return_submatrix_coords)
    """
    N = len(vars)
    K = len(perturbations)
    which_layer = index // 10
    ind = index % 10
    v = list(vars)
    v_layer = v.pop(which_layer)

    shape_matrix = np.zeros((H, W, N), np.dtype('float'))
    for i in range(N - 1):
        vertex = np.array(v[i][0:6])
        args = vertex, np.max(v[i][-1], 0)
        shape_matrix[:, :, i + 1] = triag_matrix_2(args[0], args[1], H, W)

    shape_matrix[:, :, 0] = np.ones((H, W), np.dtype('float')) * background_alpha  # adds white background
    alpha_sum = np.sum(shape_matrix, axis=2)  # H,W dimension.

    # The common part of the calculation.
    RGB = list(np.asarray(v)[:, 6:9])  # colors N - 1
    RGB.insert(0, background_color)  # insert manually the background
    RGB = np.asarray(RGB)
    RGB = RGB * (RGB < 1) * (RGB >= 0) + (RGB >= 1)  # Colors must be between 0 and 1.
    out = shape_matrix @ RGB  # the matrix multiplication magic

    outs = np.zeros((K, H, W, 3))
    s_x, s_y = 0, 0
    e_x, e_y = H - 1, W - 1
    for k in range(K):
        u = np.copy(v_layer)
        u[ind] += perturbations[k]
        vertex = np.asarray(u)[0:6]
        args = vertex, np.max((u[-1], 0))
        lay = triag_matrix_2(args[0], args[1], H, W)
        if return_submatrix_coords:
            x_axis, y_axis = np.nonzero(lay)
            x_min = np.min(x_axis)
            x_max = np.max(x_axis)
            y_min = np.min(y_axis)
            y_max = np.max(y_axis)
            if s_x < x_min:
                s_x = x_min
            if s_y < y_min:
                s_y = y_min
            if e_x > x_max:
                e_x = x_max
            if e_y > y_max:
                e_y = y_max
        a_sum = alpha_sum + lay
        if np.min(a_sum) <= 0:
            print('Problem!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        lay.shape = H, W, 1
        col = np.asarray(u[6:9])
        col = col * (col >= 0) * (col < 1) + (col >= 1) # Colors must be between 0 and 1.
        col.shape = 1, 3
        last = lay @ col
        outs[k] = out + last
        for i in range(3):  # Normalization
            outs[k][:, :, i] = (outs[k][:, :, i] / a_sum)
    if return_submatrix_coords:
        return outs, (s_x, e_x, s_y, e_y)
    else:
        return outs

