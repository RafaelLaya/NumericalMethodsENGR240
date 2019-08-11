import numpy as np

from differentiation import derivative
from differentiation import gradient

""" 
    Provides methods for root finding of continous real valued functions 
"""


def brents_method(func, a, b, tol=0.0001):
    """ Approximates one root (if existent) of the single variable function func on the interval [a, b]
    Using Brent's method

    Args:
      func (float -> float): A real valued of a single real variable function
      a (float): The start endpoint of the interval
      b (float): The ending point of the interval
      tol (float): The tolerance of the approximated error
    Raises:
      ValueError: Whatever func raises when func finds an invalid value (i.e. [a, b] is not a valid interval)
    Returns:
      (float): The approximated root of the function func on [a, b] (if found)
      None (if f(a) has the same sign as f(b))
    """
    counter = 0
    f_a = func(a)
    f_b = func(b)

    if f_a == 0:
        return a
    elif f_b == 0:
        return b
    elif f_a * f_b >= 0:
        return None

    if np.math.fabs(b) > np.math.fabs(a):
        temp = b
        b = a
        a = temp
        temp = f_a
        f_a = f_b
        f_b = temp

    c = a
    f_c = f_a
    has_used_bisection = True
    f_s = f_b

    while f_b != 0 and f_s != 0 and np.math.fabs(b - a) >= tol:
        counter += 1
        if f_a != f_c and f_c != f_b:
            s = (f_c * f_a * b) / ((f_b - f_c) * (f_b - f_a)) + f_b * f_a * c / (
                    (f_c - f_b) * (f_c - f_a)) + f_b * f_c * a / ((f_a - f_b) * (f_a - f_c))
            f_s = func(s)
        else:
            s = b - f_b * (b - a) / (f_b - f_a)
            f_s = func(s)

        if (s > max((3 * a + b) / 4, b) or s < min((3 * a + b) / 4, b)) \
                or (has_used_bisection and np.math.fabs(s - b) >= np.math.fabs(b - c) / 2) \
                or (not has_used_bisection and np.math.fabs(s - b) >= np.math.fabs(c - d) / 2) \
                or (has_used_bisection and np.math.fabs(b - c) < np.math.fabs(tol)) \
                or (not has_used_bisection and np.math.fabs(c - d) < np.math.fabs(tol)):
            s = (a + b) / 2
            f_s = func(s)
            has_used_bisection = True
        else:
            has_used_bisection = False

        d = c
        c = b
        f_c = f_b
        if f_b * f_s > 0:
            b = s
            f_b = f_s
        else:
            a = s
            f_a = f_s

        if np.math.fabs(f_a) < np.math.fabs(f_b):
            temp = a
            a = b
            b = temp
            temp = f_a
            f_a = f_b
            f_b = temp
    return s


def incremental_search(func, a, b, h=0.1):
    """ Approximates roots (if existent) of the single variable function func on the interval [a, b] using
    Incremental Search

        Args:
          func (float -> float): A real single real variable function
          a (float): The start endpoint of the interval
          b (float): The ending point of the interval
          h (float): The step-size of the incremental search
        Raises:
          ValueError: Whatever func raises when func finds an invalid value (i.e. [a, b] is not a valid interval)
        Returns:
          A list of the found approximated roots (floats) of the function func on [a, b] (if found)
        """
    interval = np.arange(a, b, h)
    result = []
    x_old = a
    y_old = func(a)
    for x_new in interval:
        y_new = func(x_new)
        if y_new == 0:
            result.append(x_new)
        elif y_new * y_old < 0:
            result.append((x_new + x_old) / 2)
        y_old = y_new
        x_old = x_new
    return result


def bisection(func, a, b, error=0.0001):
    """ Approximates one root (if existent) of the single variable function func on the interval [a, b] using
    bisection
        Args:
          func (float -> float): A real single real variable function
          a (float): The start endpoint of the interval
          b (float): The ending point of the interval
          error (float): The upper-cap of the estimated error
        Raises:
          ValueError: Whatever func raises when func finds an invalid value (i.e. [a, b] is not a valid interval)
        Returns:
          (float): The approximated root of the function func on [a, b] (if found)
        """
    n = np.math.ceil(np.log2((b - a) / error))
    f_a = func(a)
    f_b = func(b)
    x_i = (a + b) / 2
    for i in range(0, n, 1):
        x_i = (a + b) / 2
        y_i = func(x_i)
        if (y_i == 0):
            return x_i
        elif (y_i * f_a > 0):
            a = x_i
            f_a = y_i
        else:
            b = x_i
            f_b = y_i
    return x_i


def false_position(func, x_left, x_right, tol=0.001):
    """ Approximates one root (if existent) of the single variable function func on the interval [a, b] using
    False Position

        Args:
          func (float -> float): A real single real variable function
          x_left (float): The start endpoint of the interval
          x_right (float): The ending point of the interval
          tol (float): The tolerance of the approximated error
        Raises:
          ValueError: Whatever func raises when func finds an invalid value (i.e. [a, b] is not a valid interval)
        Returns:
          (float): The approximated root of the function func on [a, b] (if found)
        """
    error = tol + 1.0
    f_x_left = func(x_left)
    f_x_right = func(x_right)
    x_new = (x_left + x_right) / 2
    x_old = x_left
    while error >= tol:
        x_new = x_right - ((f_x_right * (x_left - x_right)) / (f_x_left - f_x_right))
        f_new = func(x_new)

        if f_new == 0:
            return x_new
        elif f_new * f_x_left > 0:
            x_left = x_new
        else:
            x_right = x_new
        error = np.math.fabs((x_new - x_old) / x_new) * 100
        x_old = x_new
    return x_new


def incremental_with_bisection(func, a, b, error=0.0001, h=0.1):
    """ Approximates roots (if existent) of the single variable function func on the interval [a, b] using
    Incremental Search and Bisection

        Args:
          func (float -> float): A real single real variable function
          a (float): The start endpoint of the interval
          b (float): The ending point of the interval
          error (float): The tolerance of the approximated error
          h (float): The step-size of the incremental search
        Raises:
          ValueError: Whatever func raises when func finds an invalid value (i.e. [a, b] is not a valid interval)
        Returns:
          A list of the found approximated roots (float) of the function func on [a, b] (if found)
        """
    intervals = __incremental_interval_detector(func, a, b, h)
    results = []
    for x_old in intervals:
        results.append(bisection(func, x_old, intervals[x_old], error))
    return results


def __incremental_interval_detector(func, a, b, h):
    """ Given a real function of a single real variable f, and an interval [a, b], this calculates the intervals
    used in Incremental Search (Suppose they are not known. Then this function will try to find them)

        Args:
          func (float -> float): A real single real variable function
          a (float): The start endpoint of the interval
          b (float): The ending point of the interval
          h (float): The step-size of the interval detection
        Raises:
          ValueError: Whatever func raises when func finds an invalid value (i.e. [a, b] is not a valid interval)
        Returns:
          A dictionary (float to float): Key is a value of the independent variable and value is another value of
            the independent variable such that their signs are different
        """
    interval = np.arange(a, b, h)
    result = {}
    x_old = a
    y_old = func(a)
    for x_new in interval:
        y_new = func(x_new)
        if y_new == 0:
            result[x_old] = x_new
        elif y_new * y_old < 0:
            result[x_old] = x_new
        y_old = y_new
        x_old = x_new
    return result


def secant_method(func, x_0, tol=1.0 / 100, h=0.0001):
    """ Calculate one the root of func
        Args:
          func (float -> float): A real single real variable function
          x_0 (float): Initial guess of the root
          tol (float): the tolerance of the estimated error
          h (float): the increment-size of the derivative
        Returns:
          (float): the estimated root
        """
    error = tol + 1
    x_n = 0
    while (error >= tol):
        x_n = x_0 - func(x_0) / derivative(func, x_0, h)

        error = np.math.fabs((x_n - x_0))
        x_0 = x_n
    return x_n


def newton_multi_var(functions, x0, tol=0.001):
    """ Calculate one the roots of the simultaneous equations represented by functions
        Args:
            functions (array of (array of floats->float)): This is a list of functions, each of which takes a vector x
                and outputs a float. For example: if it is wanted to solve the simultaneous equations:
                    x_1 ** 2 + x_1 * x_2 - 10 = 0
                    x_2 + 3 * x_1 * x_2 ** 2 - 57 = 0
                def function1(x):
                    return x[0] ** 2 + x[0] * x[1] - 10
                def function2(x):
                    return x[1] + 3 * x[0] * x[1] ** 2 - 57
                functions = [function1, function2]
                However, the initial guess x0 is an array where the first element is the guess for x_1 and the second
                element is the guess for x_2
            x0 (array of floats): the initial guess
            tol: the tolerance of the estimated error
        Returns:
          (numpy column of floats): the estimated root. the i-th value corresponds to the i-th variable
        """
    f = np.zeros((len(functions), 1))
    J = np.zeros((len(functions), len(x0)))
    x0 = np.array(x0).reshape(len(x0), 1)
    x_new = x0.copy()
    x_old = x0.copy()
    error = x_new - x_new + 1.0 + tol
    while ((np.fabs(error) >= tol).any()):
        for i in range(len(functions)):
            f[i][0] = functions[i](x_old)
            J[i, :] = gradient(functions[i], x_old)
        x_new = np.linalg.solve(J, -f + J @ x_old.reshape(len(x_old), 1))
        error = np.abs(x_new - x_old) / x_new * 100.0
        x_old = x_new
    return x_new
