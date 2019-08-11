import numpy as np

from linear_algebra import tridiagonal_solver
from roots import bisection
from roots import brents_method
from roots import incremental_with_bisection
from roots import secant_method


def r_coefficient(x, y):
    """ calculates the coefficient of determination of a linear regression
    Args:
        x (array of floats): the values of the independent variable
        y (array of floats): the values of the dependent variable. y[i] corresponds to x[i]
    Returns:
      (float): the coefficient of determination
    """
    return (len(x) * np.sum(np.multiply(x, y)) - np.sum(x) * np.sum(y)) / (
            np.sqrt(len(x) * np.sum(np.power(x, 2)) - np.sum(x) ** 2) * np.sqrt(
        len(x) * np.sum(np.power(y, 2)) - np.sum(y) ** 2))


def linear_reg(x, y):
    """ calculates the coefficients of a linear regression
    Args:
        x (array of floats): the values of the independent variable
        y (array of floats): the values of the dependent variable. y[i] corresponds to x[i]
    Returns:
      [float, float]: a_1 is the slope, a_0 the intercept. The line is y = a_0 + a_1 * x
    """
    a_1 = (len(x) * np.sum(np.multiply(x, y)) - np.sum(x) * np.sum(y)) / (
            len(x) * np.sum(np.power(x, 2)) - np.sum(x) ** 2)
    a_0 = np.mean(y) - a_1 * np.mean(x)
    return [a_1, a_0]


def linear_reg_through_origin(x, y):
    """ calculates the slope of a linear regression forced through the origin
    Args:
        x (array of floats): the values of the independent variable
        y (array of floats): the values of the dependent variable. y[i] corresponds to x[i]
    Returns:
      [float, float]: the slope. The line is y = (result of this function) * x
    """
    return [np.sum(np.multiply(x, y)) / np.sum(np.power(x, 2)), 0]


def linear_regression_F(x, y):
    """ defines a function that represents the linear regression of a dataset
    Args:
        x (array of floats): the values of the independent variable
        y (array of floats): the values of the dependent variable. y[i] corresponds to x[i]
    Returns:
      (float -> float): a function that represents the linear regression of the given dataset
    """

    def func(z):
        return poly_eval(linear_reg(x, y), z)

    return func


def poly_eval(coeff, val):
    """ evaluates a polynomial given its coefficients
    Args:
        coeff (array of floats): The coefficients of the polynomial. From coefficients of bigger powers to smaller powers
        val (float or array of floats): the value/values of the independent variable
    Returns:
      (float): the result of evaluating the polynomial of coefficients coeff at the value val
      (array of floats): if val is an array of floats. the result at position i will be the polynomial evaluated at val[i]
    Note:
        The coefficients are organized so that the polynomial is:
        p(x) = coeff[0] * x ** (len(coeff) - 1)
             + coeff[1] * x ** (len(coeff) - 2)
             + ...
             + coeff[len(coeff) - 1]            (since x**0 = 1)
    """
    result = 0.0
    n = len(coeff) - 1
    for coef in coeff:
        result = np.add(result, np.multiply(coef, np.power(val, n)))
        n = n - 1
    return result


def fit_power_model(x, y):
    """
    Args: Fits y = alpha * e ^ (beta * x) through linearization
        x (array of floats): the values of the independent variable
        y (array of floats): the values of the dependent variable. y[i] corresponds to x[i]
    Returns:
      [float: alpha, float: beta]: the parameters of the fit
    """
    lnx = np.log(x)
    lny = np.log(y)
    coeff = linear_reg(lnx, lny)
    Beta = coeff[0]
    alpha = np.exp(coeff[1])

    return [alpha, Beta]


def fit_power_model_F(x, y):
    """
    Args: Fits y = alpha * e ^ (beta * x) through linearization
        x (array of floats): the values of the independent variable
        y (array of floats): the values of the dependent variable. y[i] corresponds to x[i]
    Returns:
        (float->float): the function that represents the power model of this fit
    """
    coeff = fit_power_model(x, y)
    alpha = coeff[0]
    Beta = coeff[1]

    def func(z):
        return np.multiply(alpha, np.power(np.e, Beta * z))

    return func


def newton_interp_poly(x, y):
    """
    Args: Calculates the newton interpolating polynomial for the given dataset
        x (numpy array of floats): the values of the independent variable
        y (numpy array of floats): the values of the dependent variable. y[i] corresponds to x[i]
    Raises:
        Exception when x and y differ in length
    Returns:
      [float->float]: the newton interpolating polynomial
    """
    if (type(x) == type([])):
        x = np.array(x)
        y = np.array(x)
    x = x.reshape(max(np.shape(x)))
    y = y.reshape(max(np.shape(y)))
    if (np.shape(x) != np.shape(y)):
        raise Exception("x and y must have the same length!")
    n = len(x)
    A = np.zeros((n, n + 1))

    for i in range(n):
        A[i][0] = x[i]
        A[i][1] = y[i]

    for j in range(2, n + 1):
        for i in range(n + 1 - j):
            A[i][j] = (A[i + 1][j - 1] - A[i][j - 1]) / (A[j - 1 + i][0] - A[i][0])

    def poly(z):
        total = A[0][1]
        for i in range(1, n):
            result = A[0][i + 1]
            for k in range(i):
                result = result * (z - x[k])
            total = total + result
        return total

    return poly


def lagrange_poly(x, y):
    """
    Args: Calculates the lagrange interpolating polynomial for the given dataset
        x (numpy array of floats): the values of the independent variable
        y (numpy array of floats): the values of the dependent variable. y[i] corresponds to x[i]
    Raises:
        Exception when x and y differ in length
    Returns:
      [float->float]: the lagrange interpolating polynomial
    """
    x = x.astype(float)
    y = y.astype(float)
    x = x.reshape(max(np.shape(x)))
    y = y.reshape(max(np.shape(y)))
    if np.shape(x) != np.shape(y):
        raise Exception("x and y must have the same length")
    n = len(x)

    def poly(z):
        result = 0.0
        for i in range(1, n + 1):
            L = 1.0 * y[i - 1]
            for j in range(1, n + 1):
                if j == i:
                    continue
                L = L * (z - x[j - 1]) / (x[i - 1] - x[j - 1])
            result = result + L
        return result

    return poly


def inverse_interp(poly, val, a, b, method="bi"):
    """
    Args: Calculates the value of the independent variable such that its image is equal to val, over the interval [a, b]
        by using a given root-finding algorithm
        poly (float->float): A polynomial (or a function)
        val (float): The desired image
        a (float): left endpoint
        b (float): right endpoint
        method (String): The specified method. Bisection, incremental with bisection, brent's method, or the secant
                         method
    Returns:
      (float): The value of the independent variable whose image is val
    """

    def func(x):
        return poly(x) - val

    if method == "bi":
        return bisection(func, a, b)
    elif method == "bii":
        return incremental_with_bisection(func, a, b)
    elif method == "bre":
        return brents_method(func, a, b)
    elif method == "ne":
        return secant_method(func, a)


def __table_lookup(x, val):
    """
    Args: looks up the index of a given value within an array
        x (array of floats): It must be ordered. Contains the values of the independent variable
        val (float): one value within the array
    Raises:
        Exception when x is not in the array
    Returns:
      (int): The index of the value val in the array x
    """
    left = 0
    right = len(x) - 1
    mid = int(right / 2)

    if val > x[-1] or val < x[0]:
        raise Exception("Bad x")

    while (right > left):
        if val > x[mid]:
            left = min(right, mid + 1)
        elif val < x[mid]:
            right = mid
        else:
            return mid
        mid = int((left + right) / 2)
    return left - 1


def linear_splines(x, y):
    """
    Args: Calculates linear splines to the given dataset
        x (array of floats): the values of the independent variable
        y (array of floats): the values of the dependent variable. y[i] corresponds to x[i]
    Returns:
      (float -> float): The linear splines piece-wise function of the given dataset
      The spline also works on (array of floats -> array of floats) basis
    """

    def spline(val):
        if (type(val) != type(0.1)):
            result = []
            for number in val:
                i = __table_lookup(x, number)
                result.append(y[i] + (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (number - x[i]))
            return np.array(result)
        else:
            i = __table_lookup(x, val)
            return y[i] + (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (val - x[i])

    return spline


def cubic_splines(x, y, which="natural"):
    """
    Args: Calculates cubic splines to the given dataset
        x (array of floats): the values of the independent variable
        y (array of floats): the values of the dependent variable. y[i] corresponds to x[i]
        which (String): Specifies end conditions. "cubic" or "notKnot"
    Returns:
      (float -> float): The cubic splines piece-wise function of the given dataset
      The spline also works on (array of floats -> array of floats) basis
    """
    n = len(x)
    A = np.zeros((n, n))
    h = []
    for i in range(n - 1):
        h.append(x[i + 1] - x[i])

    if (which == "natural"):
        A[0][0] = 1
        A[n - 1][n - 1] = 1
    elif (which == "notKnot"):
        A[0][0] = h[1]
        A[0][1] = - (h[0] + h[1])
        A[0][2] = h[0]

        A[n - 1][n - 1] = h[n - 3]
        A[n - 1][n - 2] = - (h[n - 3] + h[n - 2])
        A[n - 1][n - 3] = h[n - 2]

    for i in range(1, n - 1):
        A[i][i - 1] = h[i - 1]
        A[i][i] = 2 * (h[i - 1] + h[i])
        A[i][i + 1] = h[i]

    b = np.zeros((n, 1))
    for i in range(1, n - 1):
        b[i] = 3 * (__finite_diff(x, y, i + 1) - __finite_diff(x, y, i))

    c_coef = tridiagonal_solver(A, b).reshape((n,))
    b_coef = np.zeros((n,))
    d_coef = np.zeros((n,))
    a_coef = np.zeros((n,))

    for i in range(n - 1):
        b_coef[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c_coef[i] + c_coef[i + 1]) / 3.0
        d_coef[i] = (c_coef[i + 1] - c_coef[i]) / (3 * h[i])
        a_coef[i] = y[i]

    def spline(val):
        try:
            i = __table_lookup(x, val)
            return a_coef[i] + b_coef[i] * (val - x[i]) + c_coef[i] * (val - x[i]) ** 2 + d_coef[i] * (val - x[i]) ** 3
        except:
            result = []
            for k in range(len(val)):
                i = __table_lookup(x, val[k])
                number = val[k]
                result.append(a_coef[i] + b_coef[i] * (number - x[i]) + c_coef[i] * (number - x[i]) ** 2 + d_coef[i] * (
                        number - x[i]) ** 3)
            return np.array(result)

    return spline


def clamped_splines(x, y, y0_prime, yn_prime):
    """
    Args: Calculates clamped splines to the given dataset
        x (array of floats): the values of the independent variable
        y (array of floats): the values of the dependent variable. y[i] corresponds to x[i]
        y0_prime (float): the value of the derivative at the start point
        yn_prime (float): the value of the derivative at the end point
    Returns:
      (float -> float): The cubic splines piece-wise function of the given dataset
      The spline also works on (array of floats -> array of floats) basis
    """
    n = len(x)
    A = np.zeros((n, n))
    h = []
    for i in range(n - 1):
        h.append(x[i + 1] - x[i])

    A[0][0] = 2 * h[0]
    A[0][1] = h[0]
    A[n - 1][n - 1] = 2 * h[n - 2]
    A[n - 1][n - 2] = h[n - 2]

    for i in range(1, n - 1):
        A[i][i - 1] = h[i - 1]
        A[i][i] = 2 * (h[i - 1] + h[i])
        A[i][i + 1] = h[i]

    b = np.zeros((n, 1))
    for i in range(1, n - 1):
        b[i] = 3 * (__finite_diff(x, y, i + 1) - __finite_diff(x, y, i))

    b[0] = 3 * __finite_diff(x, y, 1) - 3 * y0_prime
    b[n - 1] = 3 * yn_prime - 3 * __finite_diff(x, y, n - 1)

    c_coef = tridiagonal_solver(A, b).reshape((n,))
    b_coef = np.zeros((n,))
    d_coef = np.zeros((n,))
    a_coef = np.zeros((n,))

    for i in range(n - 1):
        b_coef[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c_coef[i] + c_coef[i + 1]) / 3.0
        d_coef[i] = (c_coef[i + 1] - c_coef[i]) / (3 * h[i])
        a_coef[i] = y[i]

    def spline(val):
        try:
            i = __table_lookup(x, val)
            return a_coef[i] + b_coef[i] * (val - x[i]) + c_coef[i] * (val - x[i]) ** 2 + d_coef[i] * (val - x[i]) ** 3
        except:
            result = []
            for k in range(len(val)):
                i = __table_lookup(x, val[k])
                number = val[k]
                result.append(
                    a_coef[i] + b_coef[i] * (number - x[i]) + c_coef[i] * (number - x[i]) ** 2 + d_coef[i] * (
                            number - x[i]) ** 3)
            return np.array(result)

    return spline


def __finite_diff(x, y, n):
    """
    Args: Calculates the finite diference of the values at position n-1 and n
        x (array of floats): the values of the independent variable
        y (array of floats): the values of the dependent variable. y[i] corresponds to x[i]
        n (int): the index that specifies which values to use for the finite difference
    Returns:
      (float): The finite difference
    """
    return (y[n] - y[n - 1]) / (x[n] - x[n - 1])


def bilinear_interp_newt(x, y, Z):
    """
    Args: Calculates a newton polynomial of bilinear interpolation to the given data
        x (array of floats): the values of the first independent variable
        y (array of floats): the values of the second independent  variable.
        Z (numpy 2d array of floats): Z[i][j] corresponds to the image at (x[i], y[j])
    Returns:
      ((float, float) -> float): The function that given a point returns its estimated image through the interpolation
      of the given dataset
    """

    def result(x_goal, y_goal):
        x_goal_row = []
        for i in range(0, np.shape(Z)[1]):
            x_goal_row.append(newton_interp_poly(np.array(x), Z[:][i])(x_goal))
        return newton_interp_poly(np.array(y), np.array(x_goal_row))(y_goal)

    return result


def bilinear_interp_cubic_splines(x, y, Z):
    """
    Args: Calculates bilinear interpolation to the given data using cubic splines
        x (array of floats): the values of the first independent variable
        y (array of floats): the values of the second independent  variable.
        Z (numpy 2d array of floats): Z[i][j] corresponds to the image at (x[i], y[j])
    Returns:
      ((float, float) -> float): The function that given a point returns its estimated image through the interpolation
      of the given dataset
    """

    def result(x_goal, y_goal):
        x_goal_row = []
        for i in range(0, np.shape(Z)[1]):
            x_goal_row.append(cubic_splines(x, Z[:][i])(x_goal))
        return cubic_splines(y, np.array(x_goal_row))(y_goal)

    return result


def bilinear_interp_linear_splines(x, y, Z):
    """
    Args: Calculates bilinear interpolation to the given data using linear splines
        x (array of floats): the values of the first independent variable
        y (array of floats): the values of the second independent  variable.
        Z (numpy 2d array of floats): Z[i][j] corresponds to the image at (x[i], y[j])
    Returns:
      ((float, float) -> float): The function that given a point returns its estimated image through the interpolation
      of the given dataset
    """

    def result(x_goal, y_goal):
        x_goal_row = []
        for i in range(0, np.shape(Z)[1]):
            x_goal_row.append(linear_splines(x, Z[:][i])(x_goal))
        return linear_splines(y, np.array(x_goal_row))(y_goal)

    return result


def bilinear_interp_lagrange(x, y, Z):
    """
    Args: Calculates a lagrange polynomial of bilinear interpolation to the given data
        x (array of floats): the values of the first independent variable
        y (array of floats): the values of the second independent  variable.
        Z (numpy 2d array of floats): Z[i][j] corresponds to the image at (x[i], y[j])
    Returns:
      ((float, float) -> float): The function that given a point returns its estimated image through the interpolation
      of the given dataset
    """

    def result(x_goal, y_goal):
        x_goal_row = []
        for i in range(0, np.shape(Z)[1]):
            x_goal_row.append(lagrange_poly(np.array(x), Z[:][i])(x_goal))
        return lagrange_poly(np.array(y), np.array(x_goal_row))(y_goal)

    return result


def bilinear_interp_four_pts(x, y, Z):
    """
    Args: Calculates bilinear interpolation to the given data using a rectangle
        x (array of floats): the values of the first independent variable
        y (array of floats): the values of the second independent  variable.
        Z (numpy 2d array of floats): Z[i][j] corresponds to the image at (x[i], y[j])
    Returns:
      ((float, float) -> float): The function that given a point returns its estimated image through the interpolation
      of the given dataset
    """

    def result(x_val, y_val):
        return 0.0 + (x_val - x[1]) * (y_val - y[1]) / (x[0] - x[1]) / (y[0] - y[1]) * Z[0][0] \
               + (x_val - x[0]) * (y_val - y[1]) / (x[1] - x[0]) / (y[0] - y[1]) * Z[1][0] \
               + (x_val - x[1]) * (y_val - y[0]) / (x[0] - x[1]) / (y[1] - y[0]) * Z[0][1] \
               + (x_val - x[0]) * (y_val - y[0]) / (x[1] - x[0]) / (y[1] - y[0]) * Z[1][1]

    return result


def bilinear_interp(x, y, Z):
    """
    Args: Calculates bilinear interpolation to the given data
        x (array of floats): the values of the first independent variable
        y (array of floats): the values of the second independent  variable.
        Z (numpy 2d array of floats): Z[i][j] corresponds to the image at (x[i], y[j])
    Returns:
      ((float, float) -> float): The function that given a point returns its estimated image through the interpolation
      of the given dataset
    """

    def func(x_val, y_val):
        result = 0.0
        for i in range(len(x)):
            for j in range(len(y)):
                term = 1.0
                for k in range(len(x)):
                    if k != i:
                        term = term * (x_val - x[k]) / (x[i] - x[k])
                    if k != j:
                        term = term * (y_val - y[k]) / (y[j] - y[k])
                term = term * Z[i][j]
                result = result + term
        return result

    return func
