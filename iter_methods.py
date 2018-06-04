import numpy as np
import cv2
import time

from gradients import *
from draw import *
from measures import general_norm_1


def get_random_start_2(N):
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
                                np.random.uniform(),  # blue
                                np.random.uniform(),  # green
                                np.random.uniform(),  # red
                                np.random.uniform()])
        else:
            vars[i] = np.array([np.random.uniform(), np.random.uniform(),
                                np.random.uniform(), np.random.uniform(),
                                np.random.uniform(), np.random.uniform(),
                                0, 0, 0, 10])
    return vars


def get_red_random_start_2(N):
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
                                0,
                                0,
                                np.random.uniform(),
                                np.random.uniform()])
        else:
            vars[i] = np.array([np.random.uniform(), np.random.uniform(),
                                np.random.uniform(), np.random.uniform(),
                                np.random.uniform(), np.random.uniform(),
                                0, 0, 0, 10])
    return vars


def simple_gradient_method(target_image, N, norm, step, max_iter, tol, _delta=.2, diff_scheme_to_use=0,
                           use_threads=False, show_progress=False):
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
    H = target_image.shape[0]
    W = target_image.shape[1]
    x_i = get_random_start_2(N)
    # x_i = get_red_random_start_2(N)
    it = 0
    while it < max_iter:
        print('computing gradient...')
        tt = time.time()
        grad = numerical_grad(x_i, norm, target_image, delta=_delta, _scheme=diff_scheme_to_use,
                              parallel=use_threads)
        print("Iteration: {0}, Elapsed time: {1}".format(it, time.time() - tt))
        # print("x: {0}".format(x_i))
        x_next = update_x(x_i, grad, step(it), color_boundaries=True, vertex_boundaries=False)
        difference = norm(draw_image_2(x_i, H, W), target_image)
        # print("Gradient: {0}".format(grad))
        print('Gradient:')
        for i in range(10):
            if i <= 5:
                print('vertex:', grad[:, i])
            else:
                print('color:', grad[:, i])
        print("Difference from target: {0}".format(difference))
        if difference < tol:
            x_i = x_next
            break
        x_i = x_next
        if show_progress:
            imagen = draw_image_2(x_i, H, W)
            cv2.imshow("Objective", imagen)
            if it % 5 == 0:
                cv2.imwrite('iter_images/simple_grad_progress{}.png'.format(str(it)),
                            np.array((imagen * 255), np.dtype(int)))
            cv2.waitKey(1)

        it += 1
    return x_i


def simple_gradient_method_parallel(target_image, N, norm_mode, step, max_iter, tol, _delta=.2,
                                    diff_scheme_to_use=0, use_threads=True, show_progress=False):
    """
    Simple gradient descent
    :param target_image: image to adjust. must have float colors
    :param norm_mode: integer, norm to compare images
    :param step: function, step to consider
    :param max_iter: integer, max number of iterations
    :param tol: double, tolerance to stop iterations
    :param _delta: double, for numerical differentiation calculation
    :param diff_scheme_to_use: {0,1,2} to be passed to numerical differentiation
    :param use_threads: boolean
    :param show_progress: boolean
    :return:
    """
    H = target_image.shape[0]
    W = target_image.shape[1]
    x_i = get_random_start_2(N)
    # x_i = get_red_random_start_2(N)
    it = 0
    while it < max_iter:
        print('computing gradient...')
        tt = time.time()
        grad = numerical_grad_2(x_i, norm_mode, target_image, delta=_delta, _scheme=diff_scheme_to_use,
                                parallel=use_threads)
        print("Iteration: {0}, Elapsed time: {1}".format(it, time.time() - tt))
        # print("x: {0}".format(x_i))
        x_next = update_x(x_i, grad, step(it), color_boundaries=True, vertex_boundaries=False)
        difference = general_norm_1(draw_image_2(x_i, H, W), target_image, norm_mode)
        # print("Gradient: {0}".format(grad))
        print('Gradient:')
        for i in range(10):
            if i <= 5:
                print('vertex:', grad[:, i])
            else:
                print('color:', grad[:, i])
        print("Difference from target: {0}".format(difference))
        if difference < tol:
            x_i = x_next
            break
        x_i = x_next
        if show_progress:
            imagen = draw_image_2(x_i, H, W)
            cv2.imshow("Objective", imagen)
            if it % 5 == 0:
                cv2.imwrite('iter_images/simple_grad_progress{}.png'.format(str(it)),
                            np.array((imagen * 255), np.dtype(int)))
            cv2.waitKey(1)
        it += 1
    return x_i


def sim_grad_stochastic_parallel(target_image, N, norm_mode, stoc_ratio, v_max, linesearch_num_steps,
                                 linesearch_step_size, max_iter, tol, _delta=.2, diff_scheme_to_use=0, use_threads=True,
                                 show_progress=False):
    """
    Simple implementation of stochastic gradient descent. Performs linesearch to get the best point.
    :param target_image:
    :param N:
    :param norm_mode:
    :param step:
    :param stoc_ratio:
    :param v_max:
    :param linesearch_num_steps:
    :param linesearch_step_size:
    :param max_iter:
    :param tol:
    :param _delta:
    :param diff_scheme_to_use:
    :param use_threads:
    :param show_progress:
    :return:
    """
    H = target_image.shape[0]
    W = target_image.shape[1]
    x_i = get_random_start_2(N)
    # x_i = get_red_random_start_2(N)
    it = 0
    while it < max_iter:
        print('computing gradient...')
        tt = time.time()
        grad = num_stochastic_grad(x_i, norm_mode, target_image, ratio_computed=stoc_ratio, hard_max=v_max,
                                   delta=_delta, _scheme=diff_scheme_to_use, parallel=use_threads)
        print("Iteration: {0}, Elapsed time: {1}".format(it, time.time() - tt))
        # x_next = update_x(x_i, grad, step(it), color_boundaries=True, vertex_boundaries=False)
        x_next = simple_line_search(target_image, H, W, norm_mode, x_i, grad,
                                    linesearch_num_steps, linesearch_step_size)
        difference = general_norm_1(draw_image_2(x_i, H, W), target_image, norm_mode)
        # print('Gradient:')
        # for i in range(10):
        #     if i <= 5:
        #         print('vertex:', grad[:, i])
        #     else:
        #         print('color:', grad[:, i])
        print("Difference from target: {0}".format(difference))
        if difference < tol:
            x_i = x_next
            break
        x_i = x_next
        if show_progress:
            imagen = draw_image_2(x_i, H, W)
            cv2.imshow("Objective", imagen)
            if it % 5 == 0:
                cv2.imwrite('iter_images/stoc_grad_progress{}.png'.format(str(it)),
                            np.array((imagen * 255), np.dtype(int)))
            cv2.waitKey(1)
        it += 1
    return x_i


def simple_line_search(target_image, H, W, norm_mode, vars, grad, num_steps, step_size):
    """
    Simple line search in a direction. Checks "num_steps" times by advancing grad * step_size
    Returns the best variables.
    :param target_image: Image to test against
    :param H: image dim
    :param W: image dim
    :param norm_mode: norm to consider. must be threads friendly
    :param vars: initial vars
    :param grad: direction to test
    :param num_steps: integer, number of steps
    :param step_size: float, size of the steps
    :return: array, best variables.
    """
    v = [update_x(vars, grad, step_size * i) for i in range(num_steps + 1)]
    with mp.Pool() as p:
        imgs = p.starmap(draw_image_2, [(v[i], H, W) for i in range(num_steps + 1)])
    with mp.Pool() as p:
        diffs = p.starmap(general_norm_1, [(target_image, imgs[i], norm_mode) for i in range(num_steps + 1)])
    best = np.argmin(diffs)
    return v[best]


def update_x(x_i, grad, step, color_boundaries=True, vertex_boundaries=False):
    """
    returns the updated x_i, checks and corrects boundaries of color and vertex positions.
    :param x_i: Nx10 array, old state
    :param grad: gradient
    :param step: float, step
    :param color_boundaries: bool
    :param vertex_boundaries: bool
    :return: array, new state
    """
    x = x_i - step * grad
    # check alpha positive
    x[:, 9:10] = x[:, 9:10] * (x[:, 9:10] >= 0)
    if color_boundaries:
        x[:, 6:9] = x[:, 6:9] * (x[:, 6:9] < 1) * (x[:, 6:9] >= 0) + (x[:, 6:9] >= 1)
    if vertex_boundaries:
        x[:, 0:6] = x[:, 0:6] * (x[:, 0:6] < 1) * (x[:, 0:6] >= 0) + (x[:, 0:6] >= 1)
    return x


# NEEDS REFACTORING
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


# NEEDS REFACTORING
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
