import numpy as np
import cv2


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


def triag_matrix_ws(x, alpha, H, W, x_0, y_0, sub_H, sub_W):
    """
    Draws the shape of a triangle in a zero matrix with the value of alpha.
    Only draws on a submatrix of shape sub_H, sub_W. Located in x_0,y_0
    :param x: 3,2 array, the coords of the vertex
    :param alpha: double, opacity
    :param H: original height
    :param W: origina width
    :param x_0: x position of window
    :param y_0: y position of window
    :param sub_H: shape of window
    :param sub_W: shape of window
    :return: sub_H,sub_W array, shape matrix
    """
    MM = np.array([[W, H],  # DO NOT DELETE!
                   [W, H],
                   [W, H]], np.dtype('float'))
    PP = np.array([[x_0, y_0],  # DO NOT DELETE!
                   [x_0, y_0],
                   [x_0, y_0]], np.dtype('int'))
    matrix = np.zeros((sub_H, sub_W), np.dtype('float'))
    if x.shape != (3, 2):
        x.shape = 3, 2
    x = x * MM
    x = x.astype(int) - PP
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
    :param H: image dimensions
    :param W: image dimensions
    :param background_color:
    :param background_alpha:
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
    RGB = RGB * (RGB < 1) * (RGB >= 0) + (RGB >= 1)  # Colors must be between 0 and 1.
    out = shape_matrix @ RGB  # the matrix multiplication magic
    for i in range(3):  # Normalization
        out[:, :, i] = (out[:, :, i] / alpha_sum)
    return out


def draw_multi_image_2(vars, H, W, index, perturbations, background_color=(1., 1., 1.),
                       background_alpha=.05):
    """
        (there is more space for optimization if the kind of the perturbated variable is taken into account)
        (there is more space for optimization if the submatrix coords take into account the channels)
    Renders N triangles with lineal combination of colors.
    Returns k images, where k is the length of 'perturbations'
    Each image has a perturbation from the perturbation array in the vars[index] (flat) coordinate.
    :param vars: N,10 array, initial variables.
    :param index: integer, position to make the perturbation
    :param perturbations: 1D array, values of the perturbations (to be added)
    :param background_color: (r,g,b), colors between 0 and 1.
    :param background_alpha: positive float. ZERO will cause error
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
    for k in range(K):
        u = np.copy(v_layer)
        u[ind] += perturbations[k]
        vertex = np.asarray(u)[0:6]
        args = vertex, np.max((u[-1], 0))
        lay = triag_matrix_2(args[0], args[1], H, W)
        a_sum = alpha_sum + lay
        lay.shape = H, W, 1
        col = np.asarray(u[6:9])
        col = col * (col >= 0) * (col < 1) + (col >= 1)  # Colors must be between 0 and 1.
        col.shape = 1, 3
        last = lay @ col
        outs[k] = out + last
        for i in range(3):  # Normalization
            outs[k][:, :, i] = (outs[k][:, :, i] / a_sum)
    return outs


def draw_multi_image_cropped(vars, H, W, index, perturbations, background_color=(1., 1., 1.),
                             background_alpha=.05):
    """
        (there is more space for optimization if the kind of the perturbated variable is taken into account)
        (there is more space for optimization if the submatrix coords take into account the channels)
    Renders N triangles with lineal combination of colors.
    Returns k images, where k is the length of 'perturbations'
    Each image has a perturbation from the perturbation array in the vars[index] (flat) coordinate.
    :param vars: N,10 array, initial variables.
    :param index: integer, position to make the perturbation
    :param perturbations: 1D array, values of the perturbations (to be added)
    :param background_color: (r,g,b), colors between 0 and 1.
    :param background_alpha: positive float. ZERO will cause error
    :return: k,H,W,3 array, k images (and ((x1,y1),(x2,y2)) conditioned on return_submatrix_coords)
    """
    N = len(vars)
    K = len(perturbations)
    which_layer = index // 10
    ind = index % 10

    x_0, x_1, y_0, y_1 = get_diff_window_efficiently_triags(vars, index, perturbations, H, W)
    sub_H, sub_W = y_1 - y_0 + 1, x_1 - x_0 + 1

    v = list(vars)
    v_layer = v.pop(which_layer)
    shape_matrix = np.zeros((sub_H, sub_W, N), np.dtype('float'))
    for i in range(N - 1):
        vertex = np.array(v[i][0:6])
        args = vertex, np.max(v[i][-1], 0)
        shape_matrix[:, :, i + 1] = triag_matrix_ws(args[0], args[1], H, W, x_0, y_0, sub_H, sub_W)

    shape_matrix[:, :, 0] = np.ones((sub_H, sub_W), np.dtype('float')) * background_alpha  # adds white background
    alpha_sum = np.sum(shape_matrix, axis=2)  # H,W dimension.

    # The common part of the calculation.
    RGB = list(np.asarray(v)[:, 6:9])  # colors N - 1
    RGB.insert(0, background_color)  # insert manually the background
    RGB = np.asarray(RGB)
    RGB = RGB * (RGB < 1) * (RGB >= 0) + (RGB >= 1)  # Colors must be between 0 and 1.
    out = shape_matrix @ RGB  # the matrix multiplication magic

    outs = np.zeros((K, sub_H, sub_W, 3))
    for k in range(K):
        u = np.copy(v_layer)
        u[ind] += perturbations[k]
        vertex = np.asarray(u)[0:6]
        args = vertex, np.max((u[-1], 0))
        lay = triag_matrix_ws(args[0], args[1], H, W, x_0, y_0, sub_H, sub_W)
        a_sum = alpha_sum + lay
        lay.shape = sub_H, sub_W, 1
        col = np.asarray(u[6:9])
        col = col * (col >= 0) * (col < 1) + (col >= 1)  # Colors must be between 0 and 1.
        col.shape = 1, 3
        last = lay @ col
        outs[k] = out + last
        for i in range(3):  # Normalization
            outs[k][:, :, i] = (outs[k][:, :, i] / a_sum)
    return outs


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
    x_min = np.max((0, x_min))
    x_max = np.min((W, x_max))
    y_min = np.max((0, y_min))
    y_max = np.min((H, y_max))
    return x_min, x_max, y_min, y_max
