import numpy as np
import cv2
import time

from gradients import *
from draw import *
from measures import general_norm_1, get_area_of_triangle


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


def simple_gradient_method_parallel(target_image, N, norm_mode, step, max_iter, tol=1e-4, initial_x='', _delta=.2,
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
    if initial_x == '' or initial_x is None:
        x_i = get_random_start_2(N)
    else:
        x_i = np.load(initial_x)
    # x_i = get_red_random_start_2(N)
    it = 0
    objective_function = []
    times = []
    gradients = []
    start_time = time.time()
    while it < max_iter:
        print('computing gradient...')
        tt = time.time()

        index = np.arange(10 * N, dtype=int)
        grad = num_grad_best(x_i, norm_mode, target_image, index, delta=_delta, scheme=diff_scheme_to_use,
                             warm_start=True, parallel=use_threads)

        print("Iteration: {0}, Elapsed time: {1}".format(it, time.time() - tt))
        x_next = update_x(x_i, grad, step(it), color_boundaries=True, vertex_boundaries=False)
        difference = general_norm_1(draw_image_2(x_i, H, W), target_image, norm_mode)

        print("Difference from target: {0}".format(difference))
        objective_function.append(difference)
        gradients.append(grad)
        times.append(time.time() - start_time)
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
    np.save('current_OF.npy', objective_function)
    np.save('current_grads.npy', gradients)
    np.save('times.npy', times)
    return x_i


def sim_grad_stochastic_parallel(target_image, N, norm_mode, initial_x='', stoc_ratio=.1, v_max=100,
                                 linesearch_num_steps=20, linesearch_step_size=.5, max_iter=2000, tol=1e-4, _delta=.2,
                                 diff_scheme_to_use=0, null_triag_correction=False, steps_tolerance=2,
                                 triag_area_tol=.01, use_threads=True, show_progress=False, choose_triags=False):
    """
    Simple implementation of stochastic gradient descent. Performs linesearch to get the best point.
    If method gets stuck, randomizes small triangles. The stochastic part of the gradient is simply to choose
    a part of the variables (or triangles) to perform the partial derivates. The rest are set to zero.
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
    :param null_triag_correction: boolean
    :param steps_tolerance: integer
    :param triag_area_tol: float, tolerance to randomize triangle.
    :param use_threads:
    :param show_progress:
    :return:
    """
    H = target_image.shape[0]
    W = target_image.shape[1]
    if initial_x == '' or initial_x is None:
        x_i = get_random_start_2(N)
    else:
        x_i = np.load(initial_x)
    it = 0
    stuck_counter = 0
    objective_function = []
    times = []
    gradients = []
    start_time = time.time()
    while it < max_iter:
        print('computing gradient...')
        tt = time.time()

        num_vars = N * 10
        if choose_triags:
            num_triags = N
            if v_max > 0:
                size = int(np.min((np.ceil(num_triags * stoc_ratio), v_max)))
            else:
                size = int(np.ceil(num_triags * stoc_ratio))
            triag_choice = np.sort(np.random.choice(np.arange(num_triags), size, False))
            choice = np.array([10 * i + j for i in triag_choice for j in range(10)])
        else:
            if v_max > 0:
                size = np.min((round(num_vars * stoc_ratio), v_max))
            else:
                size = round(num_vars * stoc_ratio)
            choice = np.sort(np.random.choice(np.arange(num_vars), size, False))

        grad = num_grad_best(x_i, norm_mode, target_image, choice, delta=_delta, scheme=diff_scheme_to_use,
                             warm_start=True, parallel=use_threads)

        # grad = num_stochastic_grad(x_i, norm_mode, target_image, ratio_computed=stoc_ratio, hard_max=v_max,
        #                            delta=_delta, _scheme=diff_scheme_to_use, parallel=use_threads,
        #                            choose_triags=choose_triags)
        print('Gradient time: {}'.format(time.time() - tt))
        ttt = time.time()
        # x_next = update_x(x_i, grad, step(it), color_boundaries=True, vertex_boundaries=False)
        x_next = simple_line_search(target_image, H, W, norm_mode, x_i, grad,
                                    linesearch_num_steps, linesearch_step_size)
        difference = general_norm_1(draw_image_2(x_i, H, W), target_image, norm_mode)
        print('Linesearch time: {}'.format(time.time() - ttt))
        print("Iteration: {0}, Elapsed time: {1}".format(it, time.time() - tt))
        # print('Gradient:')
        # for i in range(10):
        #     if i <= 5:
        #         print('vertex:', grad[:, i])
        #     else:
        #         print('color:', grad[:, i])
        print("Difference from target: {0}".format(difference))
        objective_function.append(difference)
        gradients.append(grad)
        times.append(time.time() - start_time)
        if difference < tol:
            x_i = x_next
            break
        if null_triag_correction:
            if np.count_nonzero(x_i != x_next) == 0:
                stuck_counter += 1
                if stuck_counter >= steps_tolerance:
                    x_next = randomize_null_triags(x_next, triag_area_tol)
                    stuck_counter = 0
            else:
                stuck_counter = 0
        x_i = x_next
        if show_progress:
            imagen = draw_image_2(x_i, H, W)
            cv2.imshow("Objective", imagen)
            if it % 5 == 0:
                cv2.imwrite('iter_images/stoc_grad_progress{}.png'.format(str(it)),
                            np.array((imagen * 255), np.dtype(int)))
            if it % 50 == 0:
                np.save('iter_vars/stoc_grad_vars{}.npy'.format(str(it)), x_i)
            cv2.waitKey(1)
        it += 1
    np.save('current_OF.npy', objective_function)
    np.save('current_grads.npy', gradients)
    np.save('times.npy', times)
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


def randomize_null_triags(v, tolerance=.005):
    """
    Checks the set of variables defining triangles and randomizes the triangles with an area below tolerance.
    :param v: array N,10. variables
    :param tolerance: float. tolerance for small triangles.
    :return: array, modified variables.
    """
    for i in range(len(v)):
        triag = v[i][0:6]
        if get_area_of_triangle(triag) < tolerance:
            v[i] = update_x(v[i].reshape(1,10), np.random.uniform(-1, 1, 10), 0.05)
            # v[i] += 0.5 * np.array([np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1),
            #                   np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1),
            #                   np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1),
            #                   np.random.uniform(-0.1,0.1),  # blue
            #                   np.random.uniform(-0.1,0.1),  # green
            #                   np.random.uniform(-0.1,0.1),  # red
            #                   np.random.uniform(-0.1,0.1)])
    return v


def accelerated_descent(target_image, N, norm_mode, L, theta_mode, initial_x='', max_iter=2000, tol=1e-4, delta=.2,
                        diff_scheme_to_use=0, use_threads=True, show_progress=False):
    """
    Accelerated descent scheme.
    :param target_image: array
    :param N: int
    :param norm_mode: int
    :param L: positive float, the lipchitz constant.
    :param theta_mode: int, 0: theta_1 = 2/(2+i). 1: theta_i+1 = 2/(1 + sqrt(1 + 4/theta_i))
    :param max_iter: int
    :param tol: float
    :param delta: step for the numerical derivative.
    :param diff_scheme_to_use: int, scheme for the numerical derivative
    :param use_threads: bool
    :param show_progress: bool
    :return:
    """
    """
    x_i, z_i
    y_i = (1-theta_i)x_i + theta_i * z_i
    g = grad(y_i)
    z_i+1 = z_i - g/(theta_i * L) 
    x_i+1 = (1 - theta_i)x_i + theta_i z_i+1
    
    theta_i+1 = 2/(1 + sqrt(1 + 4/theta_i))
    """
    H = target_image.shape[0]
    W = target_image.shape[1]
    if initial_x == '' or initial_x is None:
        x_i = get_random_start_2(N)
    else:
        x_i = np.load(initial_x)
    z_i = np.copy(x_i)
    it = 1
    th = 1
    objective_function = []
    times = []
    gradients = []
    start_time = time.time()
    while it < max_iter:
        if theta_mode == 0:
            th = 2 / (2 + it)
        else:
            th = 2 / (1 + np.sqrt(1 + 4 / th))

        y_i = (1 - th) * x_i + th * z_i

        print('computing gradient...')
        tt = time.time()

        index = np.arange(10 * N, dtype=int)
        grad = num_grad_best(x_i, norm_mode, target_image, index, delta=delta, scheme=diff_scheme_to_use,
                             warm_start=True, parallel=use_threads)

        print("Iteration: {0}, Elapsed time: {1}".format(it, time.time() - tt))

        z_next = update_x(z_i, grad, 1 / (th * L), color_boundaries=True, vertex_boundaries=False)
        x_next = (1 - th) * x_i + th * z_next

        difference = general_norm_1(draw_image_2(x_i, H, W), target_image, norm_mode)
        print("Difference from target: {0}".format(difference))
        objective_function.append(difference)
        gradients.append(grad)
        times.append(time.time() - start_time)
        if difference < tol:
            x_i = x_next
            z_i = z_next
            break
        x_i = x_next
        z_i = z_next
        if show_progress:
            imagen = draw_image_2(x_i, H, W)
            cv2.imshow("Objective", imagen)
            if it % 5 == 0:
                cv2.imwrite('iter_images/accelerated_grad_progress{}.png'.format(str(it)),
                            np.array((imagen * 255), np.dtype(int)))
            cv2.waitKey(1)
        it += 1
    np.save('current_OF.npy', objective_function)
    np.save('current_grads.npy', gradients)
    np.save('current_times.npy', times)
    return x_i


def greedy_descent(target_image, N, norm_mode, initial_x='',
                   linesearch_num_steps=40, linesearch_step_size=.25, max_iter=2000, tol=1e-4, _delta=.2,
                   diff_scheme_to_use=0, null_triag_correction=False, steps_tolerance=2,
                   triag_area_tol=.01, use_threads=True, show_progress=False, choose_triags=False):

    H = target_image.shape[0]
    W = target_image.shape[1]
    if initial_x == '':
        x_i = get_random_start_2(N)
    else:
        x_i = np.load(initial_x)
    it = 0
    stuck_counter = 0
    objective_function = []
    times = []
    gradients = []
    start_time = time.time()
    while it < max_iter:
        print('computing gradient...')
        tt = time.time()

        index = np.arange(10 * N, dtype=int)
        grad = num_grad_best(x_i, norm_mode, target_image, index, delta=_delta, scheme=diff_scheme_to_use,
                             warm_start=True, parallel=use_threads)

        num_vars = N * 10
        if choose_triags:
            norms = []
            for i in range(N):
                norms.append(np.linalg.norm(grad[i]))
                index_max = np.argmax(norms)
            for i in range(N):
                if i != index_max:
                    grad[i] = np.zeros_like(grad[i])
        else:
            grad = np.ravel(grad)
            index_max = np.argmax(grad)
            v = grad[index_max]
            grad = np.zeros_like(grad)
            grad[index_max] = v
            grad = grad.reshape(N,10)

        print('Gradient time: {}'.format(time.time() - tt))
        ttt = time.time()
        # x_next = update_x(x_i, grad, step(it), color_boundaries=True, vertex_boundaries=False)
        x_next = simple_line_search(target_image, H, W, norm_mode, x_i, grad,
                                    linesearch_num_steps, linesearch_step_size)
        difference = general_norm_1(draw_image_2(x_i, H, W), target_image, norm_mode)
        print('Linesearch time: {}'.format(time.time() - ttt))
        print("Iteration: {0}, Elapsed time: {1}".format(it, time.time() - tt))
        # print('Gradient:')
        # for i in range(10):
        #     if i <= 5:
        #         print('vertex:', grad[:, i])
        #     else:
        #         print('color:', grad[:, i])
        print("Difference from target: {0}".format(difference))
        objective_function.append(difference)
        gradients.append(grad)
        times.append(time.time() - start_time)
        if difference < tol:
            x_i = x_next
            break
        if null_triag_correction:
            if np.count_nonzero(x_i != x_next) == 0:
                stuck_counter += 1
                if stuck_counter >= steps_tolerance:
                    x_next = randomize_null_triags(x_next, triag_area_tol)
                    stuck_counter = 0
            else:
                stuck_counter = 0
        x_i = x_next
        if show_progress:
            imagen = draw_image_2(x_i, H, W)
            cv2.imshow("Objective", imagen)
            if it % 5 == 0:
                cv2.imwrite('iter_images/greedy_grad_progress{}.png'.format(str(it)),
                            np.array((imagen * 255), np.dtype(int)))
            if it % 50 == 0:
                np.save('iter_vars/greedy_grad_vars{}.npy'.format(str(it)), x_i)
            cv2.waitKey(1)
        it += 1
    np.save('current_OF.npy', objective_function)
    np.save('current_grads.npy', gradients)
    np.save('current_times.npy', times)
    return x_i
