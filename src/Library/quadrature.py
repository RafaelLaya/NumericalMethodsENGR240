import numpy as np

""" 
    Provides methods for calculating integrals (single and double) of real valued functions. Some terms used in this file:
    * Open Quadrature: When the algorithm avoids using the endpoints in order to calculate the integral.
    * Cumulative: Returns a list or array of the intermediate values with the final result at the end of the list.
    * Adaptive: As conventionally used.
    * Composite rules: As conventionally used.  
    
"""


def trapezoidal_rule(a, b, f_a, f_b):
    """ Approximates the integral of a function that goes through (a, f_a) and (b, f_b) over the interval [a, b]
    using a single interval and the trapezoidal rule

    Args:
        a (float): Starting point of the interval
        b (float): Ending point of the interval
        f_a (float): Image (y-value) of a
        f_b (float): y-value of b
    Returns:
      (float): The approximated integral of a function that goes through (a, f_a) and (b, f_b) over [a, b]
    """
    return (b - a) * (f_a + f_b) / 2.0


def trapezoidal_composite(x, y, unequal=False):
    """ Approximates the integral of a function that goes through (x_0, y_0), ..., (x_4, y_4) over the interval [x_0, x_4]
    using a single application of boole's 5th rule

    Args:
        x: An array (floats) containing the x-values of a function
        y: the images (floats) of the x-values. y[i] corresponds to x[i]. i.e. the point (x[i], y[i]) is on the graph of a function
    Returns:
      (float): The approximated value of the integral
    """
    if not unequal:
        return (x[-1] - x[0]) / (2.0 * (len(x) - 1)) * ((y[0]) + 2 * np.sum(y[1:len(y) - 1]) + (y[-1]))
    else:
        result = []
        for i in range(len(x) - 1):
            result.append(trapezoidal_rule(x[i], x[i + 1], y[i], y[i + 1]))
        return np.sum(np.array(result))


def trapezoidal_cumulative(x, y):
    """ Approximates the integral of a function that goes through (x[i], y[i]) for all valid values of i using
    the trapezoidal method (in cumulative fashion).

    Args:
       x: An array (floats) containing the x-values of a function
       y: the images (floats) of the x-values. y[i] corresponds to x[i]. i.e. the point (x[i], y[i]) is on the graph of a function
    Returns:
      A list (floats) where result[i] has the approximated integral from x[0] to x[i]
    """
    result = []
    for i in range(len(x) - 1):
        if i > 0:
            result.append(result[-1] + trapezoidal_rule(x[i], x[i + 1], y[i], y[i + 1]))
        else:
            result.append(trapezoidal_rule(x[i], x[i + 1], y[i], y[i + 1]))
    return result


def trapezoidal_cumulative_F(func, a, b, n):
    """ Approximates the integral of a real function func of a single real variable over the interval [a, b] using n
    sub-intervals to apply the trapezoidal cumulative method

    Args:
        func (float -> float): a real function of a single real variable
        a (float): start of the interval
        b (float): end of the interval
        n (int): number of sub-intervals
    Returns:
      A list (floats) where result[i] has the approximated integral from x[0] to x[i]
    """
    result = []
    x = np.linspace(a, b, n + 1)
    for i in range(len(x) - 1):
        if i > 0:
            result.append(result[-1] + trapezoidal_rule(x[i], x[i + 1], func(x[i]), func(x[i + 1])))
        else:
            result.append(trapezoidal_rule(x[i], x[i + 1], func(x[i]), func(x[i + 1])))
    return result


def trapezoidal_composite_F(func, a, b, n):
    """ Approximates the integral of a single real function func of real variable over the interval [a, b]
    using the composite trapezoidal rule

    Args:
        a (float): Start of the interval
        b (float): end of the interval
        n (int): number of sub-intervals
    Returns:
      (float): The approximated integral
    """
    x = np.linspace(a, b, n + 1)
    return (b - a) / (2.0 * n) * (func(x[0]) + 2 * np.sum(func(x[1:len(x) - 1])) + func(x[-1]))


def simpson_one_third_rule(x_0, x_1, x_2, y_0, y_1, y_2):
    """ Approximates the integral of a function that goes through (x_0, y_0), (x_1, y_1), (x_2, y_2)
     over the interval [x_0, x_2]
    using a single application of Simpson's one third rule

    Args:
        x_0 (float): Start point of the interval
        x_1 (float): Point inside the interval
        x_2 (float): End point of the interval
        y_0 (float): image of x_0
        y_1 (float): image of x_1
        y_2 (float): image of x_2
    Returns:
      (float): The approximated integral
    """
    return (x_2 - x_0) * (y_0 + 4.0 * y_1 + y_2) / 6.0


def simpson_one_third_composite(x, y):
    """ Approximates the integral of a function that goes through (x[i], y[i]) for every defined value of x, over
    the interval x[0] to x[-1]. Uses simpson's one third composite rule

    Args:
        x: an array (floats) containing the x-values
        y: an array (floats) containing the y-values. y[i] corresponds to x[i] for every defined value of i
    Raises:
        Exception when there is an odd number of elements in x
    Returns:
      (float) The approximated integral of a function that goes through the given points
    """
    if (len(x) - 1) % 2 != 0:
        raise Exception("n must be even! thus we need uneven points")
    return (x[-1] - x[0]) / (3.0 * (len(x) - 1)) * (
            y[0] + 4 * np.sum(y[1:len(y) - 1:2]) + 2 * np.sum(y[2:len(y) - 1:2]) + y[-1])


def simpson_one_third_composite_F(func, a, b, n):
    """ Approximates the integral of a real function func of a single real variable that goes through over the interval
    [a, b]. Uses simpson's one third composite rule

    Args:
        a (float): start of the interval
        b (float): end of the interval
    Raises:
        Exception when there is an odd number of sub-intervals
    Returns:
      (float): The approximated integral of func over [a, b] with n sub-intervals
    """
    if n % 2 != 0:
        raise Exception("Need even segments!")
    x = np.linspace(a, b, n + 1)
    return (b - a) / (3.0 * n) * (
            func(x[0]) + 4 * np.sum(func(x[1:len(x) - 1:2])) + 2 * np.sum(func(x[2:len(x) - 1:2])) + func(x[-1]))


def simpson_three_eight_rule(x_0, x_1, x_2, x_3, y_0, y_1, y_2, y_3):
    """ Approximates the integral of a function that goes through (x_0, y_0), ..., (x_3, y_3), over
    the interval x_0 to x_3. Uses simpson's one three eight rule

    Args:
        x_i (i=0,...,3) (floats): x-values (input) of the function
        y_i (i=0,...,3) (floats): images of x_i values
    Returns:
      (float) The approximated integral of a function that goes through the given points
    """
    return (x_3 - x_0) / 8.0 * (y_0 + 3 * y_1 + 3 * y_2 + y_3)


def boole_rule_5(x_0, x_1, x_2, x_3, x_4, y_0, y_1, y_2, y_3, y_4):
    """ Approximates the integral of a function that goes through (x_0, y_0), ..., (x_4, y_4) over the interval [x_0, x_4]
    using a single application of boole's 5th rule

    Args:
        x_i (i=0,...,4) (floats): x-values (input) of the function
        y_i (i=0,...,4) (floats): images of x_i values
    Returns:
      (float) The approximated integral of a function that goes through the given points
    """
    return (x_4 - x_0) * (7 * y_0 + 32 * y_1 + 12 * y_2 + 32 * y_3 + 7 * y_4) / 90.0


def boole_rule_6(x_0, x_1, x_2, x_3, x_4, x_5, y_0, y_1, y_2, y_3, y_4, y_5):
    """ Approximates the integral of a function that goes through (x_0, y_0), ..., (x_5, y_5) over the interval [x_0, x_5]
    using a single application of boole's 6th rule

    Args:
        x_i (i=0,...,5) (floats): x-values (input) of the function
        y_i (i=0,...,5) (floats): images of x_i values
    Returns:
      (float) The approximated integral of a function that goes through the given points
    """
    return (x_5 - x_0) * (19 * y_0 + 75 * y_1 + 50 * y_2 + 50 * y_3 + 75 * y_4 + 19 * y_5) / 288.0


def simpson(x, y):
    """ Approximates the integral of a function that goes through (x[i], y[i]) for all valid values of i.
    Uses simpson's one 1/3 and 3/8 rules

    Args:
        x: array (floats) with the input values
        y: array (floats) containing the images of the x-values. y[i] corresponds to x[i]
    Returns:
      (float): The approximated integral
    """
    if (len(x) - 1) % 2 == 0:
        return simpson_one_third_composite(x, y)
    else:
        return simpson_three_eight_rule(x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]) + simpson_one_third_composite(
            x[3:], y[3:])


def simpson_F(func, a, b, n):
    """ Approximates the integral of a real function of a single real variable over the interval [a, b]
    Uses simpson's one 1/3 and 3/8 rules

    Args:
        func (float -> float): A real valued function of a single real variable
        a (float): start point
        b (float): end point
        m (int): number of sub-intervals
    Returns:
      (float): The approximated value of the integral
    """
    x = np.linspace(a, b, n + 1)
    y = func(x)
    return simpson(x, y)


def simpson_double(x, y, Z):
    """ Approximates the double integral of a real function of two real variables using simpson's rule over the rectangle
    [x[0], x[-1]] x [y[0], y[-1]]

    Args:
        x: vector with x-values (floats) ordered ex: 1.0, 2.0, ..., 10.0
        y: vector with y-values (floats) oredered ex: -10.0, -9.0, ..., -1.0
        Z: Z[i][j] will have the function value (floats) at (x[i]. y[j]). Ex: Z[0][0] will be the image of the point (1.0, -10.0)
        if using the examples above
    Returns:
      (float): The approximated value of the double integral
    """
    A = np.zeros(len(y), )
    for i in range(len(y)):
        A[i] = simpson(x, Z[:][i])
    return simpson(y, A)


def trapezoidal_double(x, y, Z):
    """ Approximates the double integral of a real function of two real variables using the trapezoidal rule over
     the rectangle [x[0], x[-1]] x [y[0], y[-1]]
    Args:
        x: vector with x-values (floats) ordered ex: 1.0, 2.0, ..., 10.0
        y: vector with y-values (floats) oredered ex: -10.0, -9.0, ..., -1.0
        Z: Z[i][j] will have the function value (floats) at (x[i]. y[j]). Ex: Z[0][0] will be the image of the point (1.0, -10.0)
        if using the examples above
    Returns:
      (float): The approximated value of the double integral
    """
    A = np.zeros(len(y), )
    for i in range(len(y)):
        A[i] = trapezoidal_composite(x, Z[:][i])
    return trapezoidal_composite(y, A)


def open_midpoint_2(func, a, b):
    """ Approximates the integral of a real function func of a single real variable over the open interval (a, b)
    Uses the midpoint method with n=2 partitions

    Args:
        func (float->float): A real valued function of a single real variable
        a (float): start point
        b (float): end point
    Returns:
      (float): The approximated value of the integral
    """
    return (b - a) * func((a + b) / 2.0)


def open_midpoint_3(func, a, b):
    """ Approximates the integral of a real function func of a single real variable over the open interval (a, b)
    Uses the midpoint method with n=3 partitions

    Args:
        func (float->float): A real valued function of a single real variable
        a (float): start point
        b (float): end point
    Returns:
      (float) The approximated value of the integral
    """
    x_1 = a + (b - a) / 3.0
    x_2 = a + 2.0 * (b - a) / 3.0
    return (b - a) * (func(x_1) + func(x_2)) / 2.0


def open_midpoint_4(func, a, b):
    """ Approximates the integral of a real function func of a single real variable over the open interval (a, b)
    Uses the midpoint method with n=4 partitions

    Args:
        func (float->float): A real valued function of a single real variable
        a (float): start point
        b (float): end point
    Returns:
      (float) The approximated value of the integral
    """
    x_1 = a + (b - a) / 4.0
    x_2 = a + 2.0 * (b - a) / 4.0
    x_3 = a + 3.0 * (b - a) / 4.0
    return (b - a) * (2.0 * func(x_1) - func(x_2) + 2.0 * func(x_3)) / 3.0


def open_midpoint_5(func, a, b):
    """ Approximates the integral of a real function func of a single real variable over the open interval (a, b)
    Uses the midpoint method with n=5 partitions

    Args:
        func (float->float): A real valued function of a single real variable
        a (float): start point
        b (float): end point
    Returns:
      (float) The approximated value of the integral
    """
    x_1 = a + (b - a) / 5.0
    x_2 = a + 2.0 * (b - a) / 5.0
    x_3 = a + 3.0 * (b - a) / 5.0
    x_4 = a + 4.0 * (b - a) / 5.0
    return (b - a) * (11.0 * func(x_1) + func(x_2) + func(x_3) + 11.0 * func(x_4)) / 24.0


def open_midpoint_6(func, a, b):
    """ Approximates the integral of a real function func of a single real variable over the open interval (a, b)
    Uses the midpoint method with n=6 partitions

    Args:
        func (float->float): A real valued function of a single real variable
        a (float): start point
        b (float): end point
    Returns:
      (float)The approximated value of the integral
    """
    x_1 = a + (b - a) / 6.0
    x_2 = a + 2.0 * (b - a) / 6.0
    x_3 = a + 3.0 * (b - a) / 6.0
    x_4 = a + 4.0 * (b - a) / 6.0
    x_5 = a + 5.0 * (b - a) / 6.0
    return (b - a) * (
            11.0 * func(x_1) - 14.0 * func(x_2) + 26.0 * func(x_3) - 14.0 * func(x_4) + 11.0 * func(x_5)) / 20.0


def open_quad(func, a, b, n):
    """ Approximates the integral of a real function func of a single real variable over the open interval (a, b)
    Uses a combination of the midpoint method with n=6 partitions and simpson's method (which is a combination of
    simpson's formulas) -- see simpson's function in this file for more information

    Args:
        func (float->float): A real valued function of a single real variable
        a (float): start point
        b (float): end point
        n (int): number of subintervals used in simpson's method
    Returns:
      (float) The approximated value of the integral
    """
    n = n + 20
    x = np.linspace(a, b, n + 1)
    try:
        with np.errstate(divide='ignore'):
            if func(a) == np.inf:
                raise RuntimeWarning
            return simpson(x[0:len(x) - 5], func(x[0:len(x) - 5])) + open_midpoint_6(func, x[len(x) - 5], b)
    except RuntimeWarning:
        try:
            with np.errstate(divide='ignore'):
                if func(b) == np.inf:
                    raise RuntimeWarning
                return open_midpoint_6(func, a, x[4]) + simpson(x[4:], func(x[4:]))
        except RuntimeWarning:
            return open_midpoint_6(func, a, x[4]) + simpson(x[4:len(x) - 5],
                                                            func(x[4:len(x) - 5])) + open_midpoint_6(func,
                                                                                                     x[len(x) - 5],
                                                                                                     b)


def romberg(func, a, b, n, tol=0.001, max_iter=20):
    """ Approximates the integral of a real function func of a single real variable over the interval [a, b]
    Uses romberg's method which is a combination of the trapezoidal composite method and richardon's extrapolations

    Args:
        a (float): start point
        b (float): end point
        n (int): number of sub-intervals
        tol (float): tolerance
        max_iter (int): maximum number of iterations. When this number is exceeded, tolerance is ignored
    Returns:
      (float) The approximated value of the integral
    """
    A = np.zeros((2, 2))
    error = tol + 1
    four_power = 4

    iter = 0
    A[0][0] = trapezoidal_composite_F(func, a, b, 2 * n)
    A[1][0] = trapezoidal_composite_F(func, a, b, n)
    A[0][1] = (four_power * A[0][0] - A[1][0]) / (four_power - 1)
    n = n * 2
    while error >= tol and iter < max_iter:
        four_power = 1
        iter = iter + 1
        B = np.zeros((np.shape(A)[0], 1))
        A = np.hstack((A, B))
        B = np.zeros((1, np.shape(A)[1]))
        A = np.vstack((B, A))
        n = 2 * n
        A[0][0] = trapezoidal_composite_F(func, a, b, n)
        for i in range(1, np.shape(A)[1]):
            four_power = four_power * 4
            A[0][i] = (four_power * A[0][i - 1] - A[1][i - 1]) / (four_power - 1)
        error = np.math.fabs((A[0][np.shape(A)[1] - 1] - A[0][np.shape(A)[1] - 2]) / A[0][np.shape(A)[1] - 1]) * 100.0
    return A[0][np.shape(A)[1] - 1]


def open_romberg(func, a, b, n, tol=0.001, max_iter=20):
    """ Approximates the integral of a real function func of a single real variable over the open interval (a, b)
    Uses a modified version of romberg's method. Combines open_quad and richardon's extrapolations

    Args:
        a (float): start point
        b (float): end point
        n (int): number of sub-intervals
        tol (double): tolerance
        max_iter (int): maximum number of iterations
    Returns:
      (float): The approximated value of the integral. When max_iter is met, tolerance is ignored
    """
    A = np.zeros((2, 2))
    error = tol + 1
    four_power = 4

    iter = 0
    A[0][0] = open_quad(func, a, b, 2 * n)
    A[1][0] = open_quad(func, a, b, n)
    A[0][1] = (four_power * A[0][0] - A[1][0]) / (four_power - 1)
    n = n * 2
    while error >= tol and iter < max_iter:
        four_power = 1
        iter = iter + 1
        B = np.zeros((np.shape(A)[0], 1))
        A = np.hstack((A, B))
        B = np.zeros((1, np.shape(A)[1]))
        A = np.vstack((B, A))
        n = 2 * n
        A[0][0] = open_quad(func, a, b, n)
        for i in range(1, np.shape(A)[1]):
            four_power = four_power * 4
            A[0][i] = (four_power * A[0][i - 1] - A[1][i - 1]) / (four_power - 1)
        error = np.math.fabs((A[0][np.shape(A)[1] - 1] - A[0][np.shape(A)[1] - 2]) / A[0][np.shape(A)[1] - 1]) * 100.0
    return A[0][np.shape(A)[1] - 1]


def guass_leg_2(func, a, b):
    """ Approximates the integral of a real function func of a single real variable over the interval [a, b]
    Uses gauss-legendre formulas which (in general) require a change of variables so that the interval is transformed
    T: [a, b] -> [-1, 1]
    In particular, this uses the formula with two points
    Args:
        func (float->float): a real function of a single real variable
        a (float): start point
        b (float): end point
    Returns:
      (float): The approximated value of the integral
    """

    def g(x):
        return func((b + a) / 2.0 + (b - a) / 2.0 * x) * (b - a) / 2.0

    x_0 = -1.0 / np.sqrt(3.0)
    x_1 = 1.0 / np.sqrt(3.0)
    return g(x_0) + g(x_1)


def gauss_leg_3(func, a, b):
    """ Approximates the integral of a real function func of a single real variable over the interval [a, b]
    Uses gauss-legendre formulas which (in general) require a change of variables so that the interval is transformed
    T: [a, b] -> [-1, 1]
    In particular, this uses the formula with three points
    Args:
        func (float->float): a real function of a single real variable
        a (float): start point
        b (float): end point
    Returns:
      (float): The approximated value of the integral
    """

    def g(x):
        return func((b + a) / 2.0 + (b - a) / 2.0 * x) * (b - a) / 2.0

    c_0 = 5.0 / 9.0
    c_1 = 8.0 / 9.0
    c_2 = 5.0 / 9.0
    x_0 = -np.sqrt(3.0 / 5.0)
    x_1 = 0.0
    x_2 = np.sqrt(3.0 / 5.0)
    return c_0 * g(x_0) + c_1 * g(x_1) + c_2 * g(x_2)


def gauss_leg_6(func, a, b):
    """ Approximates the integral of a real function func of a single real variable over the interval [a, b]
    Uses gauss-legendre formulas which (in general) require a change of variables so that the interval is transformed
    T: [a, b] -> [-1, 1]
    In particular, this uses the formula with six points
    Args:
        func (float->float): a real function of a single real variable
        a (float): start point
        b (float): end point
    Returns:
      (float): The approximated value of the integral
    """

    def g(x):
        return func((b + a) / 2.0 + (b - a) / 2.0 * x) * (b - a) / 2.0

    c_0 = 0.171324492379170
    c_1 = 0.360761573048139
    c_2 = 0.467913934572691
    c_3 = 0.467913934572691
    c_4 = 0.360761573048131
    c_5 = 0.171324492379170
    x_0 = -0.932469514203152
    x_1 = -0.661209386466265
    x_2 = -0.238619186083197
    x_3 = 0.238619186083197
    x_4 = 0.661209386466265
    x_5 = 0.932469514203152
    return c_0 * g(x_0) + c_1 * g(x_1) + c_2 * g(x_2) + c_3 * g(x_3) + c_4 * g(x_4) + c_5 * g(x_5)


def adaptive(func, a, b, tol=0.001):
    """ Approximates the integral of a real function func of a single real variable over the interval [a, b]
    Uses adaptive quadrature which uses simpson's 1/3 rule and richardon's extrapolations
    Args:
        func (float->float): a real function of a single real variable
        a (float): start point
        b (float): end point
        tol (float): the tolerance of the estimated error
    Returns:
      (float): The approximated value of the integral
    """
    c = (a + b) / 2.0
    return __adaptive_help(func, a, b, func(a), func(b), func(c), tol)


def __adaptive_help(func, a, b, fa, fb, fc, tol):
    """ Helper function of adaptive() that actually computes the result recursively
    Args:
        func (float->float): a real function of a single real variable
        a (float): start point
        b (float): end point
        fa, fb and fc (floats): pre-computed values of f(a), f(b) and f(c). These are necessary in case that function evaluations
        might be computationally expensive
        fa (float): image of a (func(a))
        fb (float): image of b (func(b))
        fc (float): image of c (func(c)) where c is between a and b (usually the average of a and b
        tol (float): tolerance of the estimated error of the quadrature
    Returns:
      (float): The approximated value of the integral of func over [a, b]
    """
    h = b - a
    c = (a + b) / 2.0
    f_d = func((a + c) / 2.0)
    f_e = func((c + b) / 2.0)

    q_1 = h / 6.0 * (fa + 4 * fc + fb)
    q_2 = h / 12.0 * (fa + 4 * f_d + 2 * fc + 4 * f_e + fb)

    if np.abs(q_2 - q_1) <= tol:
        return q_2 + (q_2 - q_1) / 15.0
    else:
        q_a = __adaptive_help(func, a, c, fa, fc, f_d, tol)
        q_b = __adaptive_help(func, c, b, fc, fb, f_e, tol)
        return q_a + q_b


def open_adaptive(func, a, b, n, tol=0.001):
    """ Approximates the integral of a real function func of a single real variable over the open interval (a, b)
    Uses open adaptive quadrature which uses open_quad() and richardon's extrapolations (see open_quad() in this file
    for more information)
    Args:
        func (float->float): a real function of a single real variable
        a (float): start point
        b (float): end point
        n (int): number of sub-intervals to use at next estimate of the (current) sub-interval. The algorithm will choose
        new sub-intervals to work on when it decides that n is not appropriate for that sub-interval.
        tol (float): The tolerance of the estimated error
    Returns:
      (float): The approximated value of the integral
    """
    c = (a + b) / 2.0
    return __open_adaptive_help(func, a, b, tol, n)


def __open_adaptive_help(func, a, b, tol, n):
    """ Helper function of open_adaptive() that actually computes the result recursively
    Args:
        func (float->float): a real function of a single real variable
        a (float): start point
        b (float): end point
        tol (float): tolerance of the estimated error for the quadrature
        n (int): number of sub-intervals to use in the next estimation of the current sub-interval
    Returns:
      (float): The approximated value of the integral of func over [a, b]
    """
    c = (a + b) / 2.0

    q_1 = open_quad(func, a, b, n)
    q_2 = open_quad(func, a, b, 2 * n)

    if np.abs(q_2 - q_1) <= tol:
        return q_2 + (q_2 - q_1) / 15.0
    else:
        q_a = __open_adaptive_help(func, a, c, tol, 2 * n)
        q_b = __open_adaptive_help(func, c, b, tol, 2 * n)
        return q_a + q_b
