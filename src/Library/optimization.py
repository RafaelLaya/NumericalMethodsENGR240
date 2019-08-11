import numpy as np

from differentiation import gradient
from linear_algebra import norm_euclidean

""" 
    Provides methods for local optimization of real valued functions of one or two real variables. It is assumed that
    the functions are continous. The functions must have a single respective maxima or minima within the given interval
    (when the algorithm is closed).  
    
    It is okay to use python arrays for arrays of functions
    All float arrays should be numpy arrays of proper length -- although functions that don't perform array math will
    work with python arrays. If you really need to use a python array I would suggest not to trace the code to see if
    all operations are valid (and meaningful) with python arrays, rather test an example and see if it works 
    A matrix (or an array of arrays of floats) is also a two dimensional numpy array
"""


def golden_section_max_search(func, x_low, x_high, tol=0.001):
    """ Estimates the value of the independent variable that maximizes the function func over [x_low, x_high] using
    golden search
    Args:
        func (float->float): A real valued function of a single real variable
        x_low (float): lowest value of the independent variable within the interval
        x_high (float): highest value of the independent variable within the interval
        tol (float): tolerance of the estimated error
    Returns:
      (float): the value of the independent variable that locally maximizes func over [x_low, x_high]
    """
    phi = (1 + np.sqrt(5)) / 2
    d = (phi - 1) * (x_high - x_low)
    x_1 = x_low + d
    x_2 = x_high - d
    f_x1 = func(x_1)
    f_x2 = func(x_2)
    error = tol + 1

    while error >= tol:
        if f_x2 > f_x1:
            x_high = x_1
            f_x_high = f_x1

            x_1 = x_2
            f_x1 = f_x2

            x_2 = x_high - (np.sqrt(5) - 1.0) * (x_high - x_low) / 2.0
            f_x2 = func(x_2)
            error = (2 - phi) * np.math.fabs((x_high - x_low) / x_1) * 100.0
        else:
            x_low = x_2
            f_xlow = f_x2

            x_2 = x_1
            f_x2 = f_x1

            x_1 = x_low + (np.sqrt(5) - 1.0) * (x_high - x_low) / 2.0
            f_x1 = func(x_1)

            error = (2 - phi) * np.math.fabs((x_high - x_low) / x_2) * 100.0
    return (x_low + x_high) / 2.0


def golden_section_min_search(func, x_low, x_high, tol=0.001):
    """ Estimates the value of the independent variable that minimizes the function func over [x_low, x_high] using
    golden search
    Args:
        func (float->float): A real valued function of a single real variable
        x_low (float): lowest value of the independent variable within the interval
        x_high (float): highest value of the independent variable within the interval
        tol (float): tolerance of the estimated error
    Returns:
      (float): the value of the independent variable that locally minimizes func over [x_low, x_high]
    """

    def negative_func(x):
        return - 1.0 * func(x)

    return golden_section_max_search(negative_func, x_low, x_high, tol)


def parabolic_interp_max_search(func, x_1, x_2, x_3, tol=0.001):
    """ Estimates the value of the independent variable that maximizes the function func over [x_low, x_high] using
    parabolic interpolation
    Args:
        func (float->float): A real valued function of a single real variable
        x_1 (float): lowest value of the independent variable within the interval
        x_2 (float): value somewhere in (x1, x3)
        x_3 (float): highest value of the independent variable within the interval
        tol (float): tolerance of the estimated error
    Returns:
      (float): the value of the independent variable that locally maximizes func over [x_1, x_3]
    """

    def negative_func(x):
        return - 1.0 * func(x)

    return parabolic_interp_min_search(negative_func, x_1, x_2, x_3, tol)


def parabolic_interp_min_search(func, x_1, x_2, x_3, tol=0.001):
    """ Estimates the value of the independent variable that minimizes the function func over [x_low, x_high] using
    parabolic interpolation
    Args:
        func (float->float): A real valued function of a single real variable
        x_1 (float): lowest value of the independent variable within the interval
        x_2 (float): value somewhere in (x1, x3)
        x_3 (float): highest value of the independent variable within the interval
        tol (float): tolerance of the estimated error
    Returns:
      (float): the value of the independent variable that locally minimizes func over [x_1, x_3]
    """
    f_x1 = func(x_1)
    f_x2 = func(x_2)
    f_x3 = func(x_3)
    error = tol + 1

    while error >= tol and x_1 != x_2 and x_2 != x_3:
        x_4 = x_2 - (1 / 2) * ((x_2 - x_1) ** 2 * (f_x2 - f_x3) - (x_2 - x_3) ** 2 * (f_x2 - f_x1)) / (
                (x_2 - x_1) * (f_x2 - f_x3) - (x_2 - x_3) * (f_x2 - f_x1))
        f_x4 = func(x_4)

        if x_4 > x_2:
            x_1 = x_2
            f_x1 = f_x2
            x_2 = x_4
            f_x2 = f_x4
        else:
            x_3 = x_2
            f_x3 = f_x2
            x_2 = x_4
            f_x2 = f_x4

        error = np.math.fabs(max(np.math.fabs(x_3 - x_2), np.math.fabs(x_2 - x_1)) / x_4) * 100.0
    return x_2


def brent_max_search(func, x_1, x_2, x_3, tol=0.001):
    """ Estimates the value of the independent variable that maximizes the function func over [x_low, x_high] using
    brent's method
    Args:
        func (float->float): A real valued function of a single real variable
        x_1 (float): lowest value of the independent variable within the interval
        x_2 (float): value somewhere in (x1, x3)
        x_3 (float): highest value of the independent variable within the interval
        tol (float): tolerance of the estimated error
    Returns:
      (float): the value of the independent variable that locally maximizes func over [x_1, x_3]
    """

    def negative_func(x):
        return - 1.0 * func(x)

    return brent_min_search(negative_func, x_1, x_2, x_3, tol)


def brent_min_search(func, x_1, x_2, x_3, tol=0.001):
    """ Estimates the value of the independent variable that minimizes the function func over [x_low, x_high] using
    brent's method
    Args:
        func (float->float): A real valued function of a single real variable
        x_1 (float): lowest value of the independent variable within the interval
        x_2 (float): a tentative minimum value (func(x2) < func(x1) and func(x2) < func(x3))
                    The easiest way to choose x_2 is to first bound the minimum by x_1 and x_3
                    then take small steps from (x_1 + x_3) / 2.0 to either x_1 or x_3 until you find a value that
                    works as a tentative minimum
        x_3 (float): highest value of the independent variable within the interval
        tol (float): tolerance of the estimated error
    Returns:
      (float): the value of the independent variable that locally minimizes func over [x_1, x_3]
      None when error
      None: when func(x2) > func(x3) or func(x2)>func(x1)
    """
    f_x1 = func(x_1)
    f_x2 = func(x_2)
    f_x3 = func(x_3)
    if f_x2 > f_x3 or f_x2 > f_x1:
        return None
    error = tol + 1

    while error >= tol and x_1 != x_2 and x_2 != x_3:
        x_4 = x_2 - 0.5 * ((x_2 - x_1) ** 2 * (f_x2 - f_x3) - (x_2 - x_3) ** 2 * (f_x2 - f_x1)) / (
                    (x_2 - x_1) * (f_x2 - f_x3) - (x_2 - x_3) * (f_x2 - f_x1))
        f_x4 = func(x_4)

        if x_4 > x_1 and x_4 < x_3 and f_x4 < f_x2 and (
                (x_2 - x_1) ** 2 * (f_x2 - f_x3) - (x_2 - x_3) ** 2 * (f_x2 - f_x1)) > tol:
            if x_4 > x_2:
                x_1 = x_2
                f_x1 = f_x2
                x_2 = x_4
                f_x2 = f_x4
            else:
                x_3 = x_2
                f_x3 = f_x2
                x_2 = x_4
                f_x2 = f_x4
        else:
            phi = (1 + np.sqrt(5)) / 2

            x_up = x_3
            f_xup = f_x3
            x_low = x_1
            f_xlow = f_x1

            d = (phi - 1) * (x_up - x_low)

            x_1 = x_low + d
            f_x1 = func(x_1)
            x_2 = x_up - d
            f_x2 = func(x_2)

            if f_x2 > f_x1:
                x_low = x_2
                f_xlow = f_x2
                x_2 = x_1
                f_x2 = f_x1
                d = (phi - 1) * (x_up - x_low)
                x_1 = x_low + d
                f_x1 = func(x_1)
                x_opt = x_1
                f_xopt = f_x1
            else:
                x_up = x_1
                f_xup = f_x1
                x_1 = x_2
                f_x1 = f_x2
                d = (phi - 1) * (x_up - x_low)
                x_2 = x_up - d
                f_x2 = func(x_2)
                x_opt = x_2
                f_xopt = f_x2
            x_1 = x_low
            f_x1 = f_xlow
            x_2 = x_opt
            f_x2 = f_xopt
            x_3 = x_up
            f_x3 = f_xup

        error = np.math.fabs(max(np.math.fabs(x_3 - x_2), np.math.fabs(x_2 - x_1)))
    return x_2


def gradient_descend(func, x_0, tol=0.001, gamma=0.1, beta=0.90):
    """ Estimates the value of the local minimum of the multivariable function func near x_0
        func (array of floats->float): the function that is going to be minimized
        x_0 (array of floats): the initial value/guess
        gamma (float (0, 1]: learning rate
        beta (float [0, 1]): momentum
    Returns:
      (numpy array of floats): the value of the independent variable that is estimated to locally minimizes func
    """
    x_0 = np.array(x_0).reshape(len(x_0), )
    current = x_0.copy()
    error = tol + 1.0
    old_gradient = current * 0
    while error >= tol:
        grad = gradient(func, current)
        current = current - gamma * (beta * old_gradient + (1.0 - beta) * grad)
        old_gradient = grad
        error = norm_euclidean(grad)
    return current


def gradient_ascent(func, x_0, tol=0.001, gamma=0.1, beta=0.90):
    """ Estimates the value of the local maximum of the multivariable function func near x_0
         func (array of floats->float): the function that is going to be minimized
         x_0 (array of floats): the initial value/guess
         gamma (float (0, 1]: learning rate
         beta (float [0, 1]): momentum
     Returns:
       (numpy array of floats): the value of the independent variable that is estimated to locally maximize func
     """

    def negative_func(z):
        return - 1.0 * func(z)

    return gradient_descend(negative_func, x_0, tol, gamma, beta)
