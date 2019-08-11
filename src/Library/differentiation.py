import numpy as np

""" Functions named (First/Second/Third/Fourth)_(derivative)_(forward/backward/centered) calculate the derivative of
    a given real valued function of a single real variable. The first part of the identifier refers to which order
    derivative (first, second, ...) and the third part of the identifier represents whether the increment is taken
    forward, backward, or centered (for positive h)
Args: 
    func(float->float): A real valued function of a single real variable
    x (float): the value of the independent variable where we want to take the derivative
    h (float): the increment when taking the derivative
Returns:
    (float): the estimate of the derivative of the function func at x
Notes: 
    Can also be coded by using first order formulas iteratively, but this improves performance for the most common order
    derivatives
"""


def first_derivative_forward(func, x, h=0.001):
    return (- func(x + 2.0 * h) + 4.0 * func(x + h) - 3 * func(x)) / (2.0 * h)


def second_derivative_forward(func, x, h=0.001):
    return (- func(x + 3.0 * h) + 4.0 * func(x + 2.0 * h) - 5.0 * func(x + h) + 2.0 * func(x)) / (h ** 2.0)


def third_derivative_forward(func, x, h=0.001):
    return (-3.0 * func(x + 4.0 * h) + 14.0 * func(x + 3.0 * h) - 24.0 * func(x + 2.0 * h) + 18.0 * func(
        x + h) - 5.0 * func(x)) / (2.0 * h ** 3.0)


def fourth_derivative_forward(func, x, h=0.001):
    return (-2.0 * func(x + 5.0 * h) + 11.0 * func(x + 4.0 * h) - 24.0 * func(x + 3.0 * h) + 26.0 * func(
        x + 2.0 * h) - 14.0 * func(x + h) + 3.0 * func(x)) / (h ** 4.0)


def first_derivative_backwards(func, x, h=0.001):
    return (3.0 * func(x) - 4.0 * func(x - h) + func(x - 2.0 * h)) / (2.0 * h)


def second_derivative_backwards(func, x, h=0.001):
    return (2.0 * func(x) - 5.0 * func(x - h) + 4.0 * func(x - 2.0 * h) - func(x - 3.0 * h)) / (h ** 2.0)


def third_derivative_backward(func, x, h=0.001):
    return (5.0 * func(x) - 18.0 * func(x - h) + 24.0 * func(x - 2.0 * h) - 14.0 * func(x - 3.0 * h) + 3.0 * func(
        x - 4.0 * h)) / (2.0 * h ** 3.0)


def fourth_derivative_backward(func, x, h=0.001):
    return (3.0 * func(x) - 14.0 * func(x - h) + 26.0 * func(x - 2.0 * h) - 24.0 * func(x - 3.0 * h) + 11.0 * func(
        x - 4.0 * h) - 2.0 * func(x - 5.0 * h)) / (h ** 4.0)


def first_derivative_centered(func, x, h=0.001):
    return (- func(x + 2.0 * h) + 8.0 * func(x + h) - 8.0 * func(x - h) + func(x - 2.0 * h)) / (12.0 * h)


def second_derivative_centered(func, x, h=0.001):
    return (- func(x + 2.0 * h) + 16.0 * func(x + h) - 30.0 * func(x) + 16.0 * func(x - h) - func(x - 2.0 * h)) / (
                12.0 * h ** 2.0)


def third_derivative_centered(func, x, h=0.001):
    return (- func(x + 3.0 * h) + 8.0 * func(x + 2.0 * h) - 13.0 * func(x + h) + 13.0 * func(x - h) - 8.0 * func(
        x - 2.0 * h) + func(x - 3.0 * h)) / (8.0 * h ** 3.0)


def fourth_derivative_centered(func, x, h=0.001):
    return (- func(x + 3.0 * h) + 12.0 * func(x + 2.0 * h) - 39.0 * func(x + h) + 56.0 * func(x) - 39.0 * func(
        x - h) + 12.0 * func(x - 2.0 * h) - func(x - 3.0 * h)) / (6.0 * h ** 4.0)


def derivative(func, x, h=0.001):
    """ Calculates the first derivative of func at x using increment h
    Args:
        func(float->float): the function whose derivative is desired
        x (float or array of floats): the values of the independent variable
        h (float): the increment
    Returns:
        (float): the estimated derivative of func at x
        (result: array of floats): if passed an array, result[i] will have the estimated derivative of func at x[i]
    """
    if isinstance(x, int):
        x = float(x)
    if isinstance(x, float):
        try:
            return first_derivative_centered(func, x, h)
        except:
            try:
                return first_derivative_forward(func, x, h)
            except:
                return first_derivative_backwards(func, x, h)
    else:
        result = []
        for val in x:
            result.append(derivative(func, val, h))
        return np.array(result)


def richardson_extrapolation(func, x, h=0.001):
    """ Improves accuracy of a first derivative by using richardson extrapolation
     Args:
         func(float->float): the function whose derivative is desired
         x (float, or array of floats): the value of the independent variable
         h (float): the derivative increment
     Returns:
         (float): the estimated derivative of func at x
         (result: array of floats): if passed an array, result[i] will have the estimated derivative of func at x[i]
     """
    if isinstance(x, float):
        try:
            return 16.0 / 15.0 * first_derivative_centered(func, x, 0.5 * h) - 1.0 / 15.0 * first_derivative_centered(
                func, x, h)
        except:
            try:
                return (4.0 / 3.0) * first_derivative_forward(func, x, 0.5 * h) - (
                            1.0 / 3.0) * first_derivative_forward(func, x, h)
            except:
                return (4.0 / 3.0) * first_derivative_backwards(func, x, 0.5 * h) - (
                            1.0 / 3.0) * first_derivative_backwards(func, x, h)
    else:
        result = []
        for val in x:
            result.append(richardson_extrapolation(func, val, h))
        return np.array(result)


def derivative_romberg(func, x, tol=0.001, max_iter=20, h=1.0):
    """ Calculates the first derivative of func at x using increment h and romberg's method with richardson extrapolations
    Args:
        func(float->float): the function whose derivative is desired
        x (float or array of floats): the values of the independent variable
        tol (float): the tolerance of the estimated error
        max_iter (int): the maximum number of iterations. When max_iter is hit, tolerance is ignored.
        h (float): the increment
    Returns:
        (float): the estimated derivative of func at x
        (result: array of floats): if passed an array, result[i] will have the estimated derivative of func at x[i]
    """
    if not isinstance(x, float) and not isinstance(x, int):
        result = []
        for val in x:
            result.append(derivative_romberg(func, val, tol, max_iter, h))
        return result

    if isinstance(x, int):
        x = float(x)

    A = np.zeros((2, 2))
    error = tol + 1
    fourth_power = 4

    iter = 0
    A[0][0] = derivative(func, x, h / 2.0)
    A[1][0] = derivative(func, x, h)
    A[0][1] = (fourth_power * A[0][0] - A[1][0]) / (fourth_power - 1)
    h = h / 2.0
    while error >= tol and iter < max_iter:
        fourth_power = 1
        iter = iter + 1
        B = np.zeros((np.shape(A)[0], 1))
        A = np.hstack((A, B))
        B = np.zeros((1, np.shape(A)[1]))
        A = np.vstack((B, A))
        h = h / 2.0
        A[0][0] = derivative(func, x, h)
        for i in range(1, np.shape(A)[1]):
            fourth_power = fourth_power * 4
            A[0][i] = (fourth_power * A[0][i - 1] - A[1][i - 1]) / (fourth_power - 1)
        error = np.math.fabs((A[0][np.shape(A)[1] - 1] - A[0][np.shape(A)[1] - 2]) / A[0][np.shape(A)[1] - 1]) * 100.0
    return A[0][np.shape(A)[1] - 1]


def finite_diff(x, y, n):
    """ Calculates the finite difference of (x[n-1], y[n-1]) and (x[n], y[n])
    Args:
        x (array of floats): independent variable
        y (array of floats): dependent variable
        n (int): index of the highest point between the two
    Returns:
        (float): the finite difference of (x[n-1], y[n-1]) and (x[n], y[n])
    """
    return (y[n] - y[n - 1]) / (x[n] - x[n - 1])


def __derivative_table_lookup(x, val):
    """ finds the index of a given value. Used as a helper of derivative_lagrange_poly()
        x (array of floats): independent variable
        val (float): the value whose index is desired
    Returns:
        (int): the index of val in x
    """
    left = 0
    right = len(x) - 1
    mid = int(right / 2)

    if val > x[-1] or val < x[0]:
        raise Exception("Check format of x")

    while (right > left):
        if val > x[mid]:
            left = min(right, mid + 1)
        elif val < x[mid]:
            right = mid
        else:
            return mid
        mid = int((left + right) / 2)
    return left - 1


def derivative_lagrange_poly(x, y, val):
    """ Given a dataset of dependent and independent variables, this will estimate the derivative at a point using
    lagrange's interpolation polynomial
    Args:
        x (array of floats): independent variable values
        y (array of floats): dependent variable values
        val (float): the point where the derivative is desired
    Returns:
        (float): the estimated derivative of func at x
        (result: array of floats): if passed an array, result[i] will have the estimated derivative of func at x[i]
    """
    if (isinstance(val, int)):
        val = float(val)
    if isinstance(val, float):
        if len(x) < 3 or len(y) < 3 or val > max(x) or val < min(x):
            raise Exception("Data out of bounds or not enough data points")

    if isinstance(val, float):
        index = __derivative_table_lookup(x, val)

        if np.abs(val - x[index + 1]) < np.abs(val - x[index]) and index + 1 < len(x):
            index = index + 1
        elif np.abs(val - x[index - 1]) < np.abs(val - x[index]) and index > 0:
            index = index - 1

        xValues = np.zeros((3,))
        yValues = np.zeros((3,))
        if index == 0:
            xValues[0] = x[0]
            xValues[1] = x[1]
            xValues[2] = x[2]
            yValues[0] = y[0]
            yValues[1] = y[1]
            yValues[2] = y[2]
        elif index == len(x) - 1:
            xValues[0] = x[len(x) - 3]
            xValues[1] = x[len(x) - 2]
            xValues[2] = x[len(x) - 1]
            yValues[0] = y[len(y) - 3]
            yValues[1] = y[len(y) - 2]
            yValues[2] = y[len(y) - 1]
        else:
            xValues[0] = x[index - 1]
            xValues[1] = x[index]
            xValues[2] = x[index + 1]
            yValues[0] = y[index - 1]
            yValues[1] = y[index]
            yValues[2] = y[index + 1]
        return yValues[0] * (2 * val - xValues[1] - xValues[2]) / (
                    (xValues[0] - xValues[1]) * (xValues[0] - xValues[2])) \
               + yValues[1] * (2 * val - xValues[0] - xValues[2]) / (
                           (xValues[1] - xValues[0]) * (xValues[1] - xValues[2])) \
               + yValues[2] * (2 * val - xValues[0] - xValues[1]) / (
                           (xValues[2] - xValues[0]) * (xValues[2] - xValues[1]))
    else:
        result = []
        for number in val:
            result.append(derivative_lagrange_poly(x, y, number))
        return np.array(result)


def gradient(func, x):
    """ Calculates the gradient of a multi variable function
        func((array of floats)->float): the function whose gradient is desired
        x (array of floats): the values of the independent variable
    Returns:
        (array of floats): the gradient of func at x
    """
    m = len(x)
    grad = np.zeros((1, m)).reshape(m, )
    for i in range(m):
        grad[i] = partialDerivative(func, x, i)
    return grad


def gradient_f(func):
    """ Calculates the gradient function of a multi variable function
        func((array of floats)->float): the function whose gradient is desired
    Returns:
        ((array of floats)->float): the gradient function of func
    """

    def g(x):
        return gradient(func, x)

    return g


def partialDerivative(func, x, i):
    """ Calculates the partial derivative of the variable with index i of the function func, at x
        func((array of floats)->float): the function whose partial derivative is desired
        x (array of floats): the values of the independent variable where the derivative is desired
        i (int): the index of the independent variable for the partial derivative
    Returns:
        (array of floats->float): the partial derivative of the function func at x (of the (i+1)th variable)
    """
    h = 0.0001
    def g(z):
        x_incr = x.copy()
        x_incr[i] = z
        return func(x_incr)

    return derivative(g, x[i], h)


def deriv_n(f, x, n, h=0.001):
    """ Calculates the nth derivative of func at x using increment h
    Args:
        func(float->float): the function whose derivative is desired
        x (float or array of floats): the values of the independent variable
        h (float): the increment
        n (float): the derivative order
    Returns:
        (float): the estimated derivative of func at x
        (result: array of floats): if passed an array, result[i] will have the estimated derivative of func at x[i]
    """
    if (n <= 1):
        return derivative(f, x, h=h)
    else:
        return deriv_n(lambda x: derivative(f, x, h=h), x, n - 1, h=h)




