import numpy as np
from skimage.measure import compare_ssim as ssim
from scipy.ndimage.measurements import center_of_mass


def general_norm_1(i1, i2, mode=0):
    """
    Computes a norm, choosing it via parameters
    mode =
    0: MSE
    1: MSE_centroid
    2: MSE_centroid_colormeans
    -1: MAE
    -2: SSIM
    :param i1: image 1
    :param i2: image 2
    :param mode: integer
    :return: float.
    """
    if mode == 0:
        return MSE(i1, i2)
    elif mode == 1:
        return MSE_w_Centroid(i1, i2)
    elif mode == 2:
        return MSE_centroid_colormeans(i1, i2)
    elif mode == -1:
        return MAE(i1, i2)
    elif mode == -2:
        return SSIM(i1, i2)
    elif mode == -3:
        return .5 * MSE(i1, i2) + .5 * SSIM(i1, i2)
    else:
        return MSE(i1, i2)


def MAE(imageA, imageB):
    """
     # Error absoluto medio (Se asume que ambas imágenes tienen el mismo tamaño)
    :param imageA: image 1
    :param imageB:  image 2
    :return: positive float
    """
    err = np.sum(np.abs(imageA.astype("float") - imageB.astype("float")))
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def MSE(imageA, imageB):
    """
    Error cuadrático medio (Se asume que ambas imágenes tienen el mismo tamaño)
    :param imageA:
    :param imageB:
    :return:
    """
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def SE(i1, i2):
    """
    Returns the Square error.
    :param i1: image 1
    :param i2: image 2
    :return:
    """
    err = np.sum((i1.astype("float") - i2.astype("float")) ** 2)
    return err


def SSIM(i1, i2):
    """
    SSIM, 1 es best.
    :param i1:
    :param i2:
    :return:
    """
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
    H = i1.shape[0]
    W = i1.shape[1]
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
    H = i1.shape[0]
    W = i1.shape[1]

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
