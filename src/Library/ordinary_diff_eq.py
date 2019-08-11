import numpy as np

"""
        Different methods for solving initial value problems in real variables. Methods for one equation or simultaneous
        (systems) of differential equations are provided
"""


def euler(func, t_0, t_f, y_0, h):
    """ Approximates the solution to the differential equation given by dy/dt=func(t, y) over the interval [t0, tf]
    with initial conditions (t0, y0) using euler's method with step-size h
    Args:
        func ((float, float)->float): This is the derivative of the dependent variable with respect to the single independent
        variable
        t_0 (float): the value of the single independent variable at the initial condition
        t_f (float): the final desired value
        y_0 (float): the value of the dependent variable at the initial condition
        h (float): the step-size of euler's method (use negative to go backwards in t)
    Returns:
     Numpy arrays (with floats) that contain the solution to dy/dt=func(t, y)
     The format is (T, Y) where Y[i] is the image of the t value given by T[i]
    """
    t = [t_0]
    y = [y_0]
    t_current = t_0 + h
    while (t_current <= t_f):
        y.append(y[-1] + h * func(t[-1], y[-1]))
        t.append(t_current)
        t_current = t_current + h
    return np.array(t), np.array(y)


def system_ode_euler(functions, t_0, t_f, y_0, h):
    """ Approximates the solution to a system of simultaneous ordinary first order differential equations of the form
    dy_i/dt = f(t, y1, ..., yn) using euler's method
    Args:
        functions ((float, array of floats) -> (float)): List of functions that represent ordinary first order
        differential equations. The ith function, called functions[i], takes two arguments. The first argument is a
        float (t) that corresponds to the a value of the independent variable. The second argument is an array that
        has the value of each dependent variable. For example: we have to solve:
            dw/dt = w + z + t
            dz/dt = w ** 2 + z ** 2 + t ** 2
            where w(0) = 5.0 and z(0) = 6.0
            w(0) corresponds to y[0] and z(0) to y[1]
            dw/dt corresponds to functions[0] and dz/dt corresponds to functions[1]
        t_0 (float): Value of independent variable at initial condition
        t_f (float): Value of independent variable at final value
        y_0 (Array of floats): values of dependent variables at initial condition. Following the example above:
            y_0 = [5.0, 6.0]
        h (float): The step-size of euler's method
    Returns:
        (T, Y) both are numpy arrays of floats. T[i] is the value of the independent variable
            Y will be a matrix where the row i will correspond to the values of the ith dependent variable.
            Y[i][j] will be the value of the ith variable at when t=T[j]
            Keep in mind these are all estimated values
    Notes:
        Remember to match arrays. First put each equation in the form dw/dt= func(t, dependent_variables). Then assign
        an integer from (0, 1, ...) to each equation. if dw/dt is assigned the number i, then functions[i] should have
        the function that describes dw/dt. y_0[i] should have the initial value of w. The result Y[i][:] will have the values
        of w. The value of w at time T[j] is Y[i][j]
    """
    t = [t_0]
    y = np.array(y_0).reshape(len(y_0), 1)
    t_current = t_0 + h
    while (t_current <= t_f):
        y_new = []
        for i in range(np.shape(y)[0]):
            y_new.append(y[i][-1] + h * functions[i](t[-1], y[:, -1]))

        y = np.hstack((y, np.array(y_new).reshape(max(np.shape(y_new)), 1)))
        t.append(t_current)
        t_current = t_current + h
    return np.array(t), y


def heun_not_iterative(func, t_0, t_f, y_0, h):
    """ Approximates the solution to the differential equation given by dy/dt=func(t, y) over the interval [t0, tf]
    with initial conditions (t0, y0) using heun's method with step-size h
    Args:
        func ((float, float)->float): This is the derivative of the dependent variable with respect to the single independent
        variable
        t_0 (float): the value of the single independent variable at the initial condition
        t_f (float): the final desired value
        y_0 (float): the value of the dependent variable at the initial condition
        h (float): the step-size of euler's method (use negative to go backwards in t)
    Returns:
     Numpy arrays (with floats) that contain the solution to dy/dt=func(t, y)
     The format is (T, Y) where Y[i] is the image of the t value given by T[i]
    """
    t = [t_0]
    y = [y_0]
    t_current = t_0 + h
    while (t_current <= t_f):
        y_approx = y[-1] + h * func(t[-1], y[-1])
        y.append(y[-1] + h / 2.0 * (func(t[-1], y[-1]) + func(t[-1] + h, y_approx)))
        t.append(t_current)
        t_current = t_current + h
    return np.array(t), np.array(y)


def heun_iterative(func, t_0, t_f, y_0, h, tol=0.001):
    """ Approximates the solution to the differential equation given by dy/dt=func(t, y) over the interval [t0, tf]
    with initial conditions (t0, y0) using a modified version of heun's method with step-size h
    Args:
        func ((float, float)->float): This is the derivative of the dependent variable with respect to the single independent
        variable
        t_0 (float): the value of the single independent variable at the initial condition
        t_f (float): the final desired value
        y_0 (float): the value of the dependent variable at the initial condition
        h (float): the step-size of euler's method (use negative to go backwards in t)
    Returns:
     Numpy arrays (with floats) that contain the solution to dy/dt=func(t, y)
     The format is (T, Y) where Y[i] is the image of the t value given by T[i]
    """
    t = [t_0]
    y = [y_0]
    t_current = t_0 + h
    while (t_current <= t_f):
        error = tol + 1.0
        y_old = y[-1] + h * func(t[-1], y[-1])
        while (error >= tol):
            y_new = y[-1] + h / 2.0 * (func(t[-1], y[-1]) + func(t[-1] + h, y_old))
            error = np.abs((y_new - y_old) / y_new) * 100
            y_old = y_new
        yApprox = y_new
        y.append(y[-1] + h / 2.0 * (func(t[-1], y[-1]) + func(t[-1] + h, yApprox)))
        t.append(t_current)
        t_current = t_current + h
    return np.array(t), np.array(y)


def midpoint_method(func, t_0, t_f, y_0, h):
    """ Approximates the solution to the differential equation given by dy/dt=func(t, y) over the interval [t0, tf]
    with initial conditions (t0, y0) using the midpoint method with step-size h
    Args:
        func ((float, float)->float): This is the derivative of the dependent variable with respect to the single independent
        variable
        t_0 (float): the value of the single independent variable at the initial condition
        t_f (float): the final desired value
        y_0 (float): the value of the dependent variable at the initial condition
        h (float): the step-size of euler's method (use negative to go backwards in t)
    Returns:
     Numpy arrays (with floats) that contain the solution to dy/dt=func(t, y)
     The format is (T, Y) where Y[i] is the image of the t value given by T[i]
    """
    t = [t_0]
    y = [y_0]
    t_current = t_0 + h
    while (t_current <= t_f):
        y_mid = y[-1] + h / 2.0 * func(t[-1], y[-1])
        y.append(y[-1] + h * func(t[-1] + h / 2.0, y_mid))
        t.append(t_current)
        t_current = t_current + h
    return np.array(t), np.array(y)


def ralston_method(func, t_0, t_f, y_0, h):
    """ Approximates the solution to the differential equation given by dy/dt=func(t, y) over the interval [t0, tf]
    with initial conditions (t0, y0) using  ralston's method
    Args:
        func ((float, float)->float): This is the derivative of the dependent variable with respect to the single independent
        variable
        t_0 (float): the value of the single independent variable at the initial condition
        t_f (float): the final desired value
        y_0 (float): the value of the dependent variable at the initial condition
        h (float): the step-size of euler's method (use negative to go backwards in t)
    Returns:
     Numpy arrays (with floats) that contain the solution to dy/dt=func(t, y)
     The format is (T, Y) where Y[i] is the image of the t value given by T[i]
    """
    t = [t_0]
    y = [y_0]
    t_current = t_0 + h
    while (t_current <= t_f):
        k1 = func(t[-1], y[-1])
        k2 = func(t[-1] + 3.0 * h / 4.0, y[-1] + 3.0 / 4.0 * k1 * h)
        y.append(y[-1] + (k1 / 3.0 + 2 * k2 / 3.0) * h)
        t.append(t_current)
        t_current = t_current + h
    return np.array(t), np.array(y)


def runge_kutta_4(func, t_0, t_f, y_0, h):
    """ Approximates the solution to the differential equation given by dy/dt=func(t, y) over the interval [t0, tf]
    with initial conditions (t0, y0) using Runge kutta's 4th order method and step-size h
    Args:
        func ((float, float)->float): This is the derivative of the dependent variable with respect to the single independent
        variable
        t_0 (float): the value of the single independent variable at the initial condition
        t_f (float): the final desired value
        y_0 (float): the value of the dependent variable at the initial condition
        h (float): the step-size of euler's method (use negative to go backwards in t)
    Returns:
     Numpy arrays (with floats) that contain the solution to dy/dt=func(t, y)
     The format is (T, Y) where Y[i] is the image of the t value given by T[i]
    """
    t = [t_0]
    y = [y_0]
    t_current = t_0 + h
    while (t_current <= t_f):
        k1 = func(t[-1], y[-1])
        k2 = func(t[-1] + h / 2.0, y[-1] + k1 * h / 2.0)
        k3 = func(t[-1] + h / 2.0, y[-1] + k2 * h / 2.0)
        k4 = func(t[-1] + h, y[-1] + k3 * h)
        y.append(y[-1] + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        t.append(t_current)
        t_current = t_current + h
    return np.array(t), np.array(y)


def system_ode_runge_kutta_4(functions, t_0, t_f, y_0, h):
    """ Approximates the solution to a system of simultaneous ordinary first order differential equations of the form
    dy_i/dt = f(t, y1, ..., yn) using Runge Kutta's fourth order method
    Args:
        functions ((float, array of floats) -> (float)): List of functions that represent ordinary first order
        differential equations. The ith function, called functions[i], takes two arguments. The first argument is a
        float (t) that corresponds to the a value of the independent variable. The second argument is an array that
        has the value of each dependent variable. For example: we have to solve:
            dw/dt = w + z + t
            dz/dt = w ** 2 + z ** 2 + t ** 2
            where w(0) = 5.0 and z(0) = 6.0
            w(0) corresponds to y[0] and z(0) to y[1]
            dw/dt corresponds to functions[0] and dz/dt corresponds to functions[1]
        t_0: Value of independent variable at initial condition
        t_f: Value of independent variable at final value
        y_0: Array of values of dependent variables at initial condition. Following the example above:
            y_0 = [5.0, 6.0]
        h: The step-size of euler's method
    Returns:
        (T, Y) both are numpy arrays of floats. T[i] is the value of the independent variable
            Y will be a matrix where the row i will correspond to the values of the ith dependent variable.
            Y[i][j] will be the value of the ith variable at when t=T[j]
            Keep in mind these are all estimated values
    Notes:
        Remember to match arrays. First put each equation in the form dw/dt= func(t, dependent_variables). Then assign
        an integer from (0, 1, ...) to each equation. if dw/dt is assigned the number i, then functions[i] should have
        the function that describes dw/dt. y_0[i] should have the initial value of w. The result Y[i][:] will have the values
        of w. The value of w at time T[j] is Y[i][j]
    """
    t = [t_0]
    y = np.array(y_0).reshape(len(y_0), 1)
    t_current = t_0 + h
    while (t_current <= t_f):
        y_new = []
        k = np.zeros((len(y_0), 4))
        y_mid = []

        for i in range(len(y_0)):
            k[i][0] = functions[i](t[-1], y[:, -1])

        for i in range(len(y_0)):
            y_mid.append(y[i, -1] + k[i][0] * h / 2.0)

        for i in range(len(y_0)):
            k[i][1] = functions[i](t[-1] + h / 2.0, y_mid)

        y_mid = []
        for i in range(len(y_0)):
            y_mid.append(y[i, -1] + k[i][1] * h / 2.0)
        for i in range(len(y_0)):
            k[i][2] = functions[i](t[-1] + h / 2.0, y_mid)

        y_mid = []
        for i in range(len(y_0)):
            y_mid.append(y[i, -1] + k[i][2] * h)
        for i in range(len(y_0)):
            k[i][3] = functions[i](t[-1] + h, y_mid)

        for i in range(len(y_0)):
            y_new.append(y[i, -1] + h / 6.0 * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]))

        y = np.hstack((y, np.array(y_new).reshape(max(np.shape(y_new)), 1)))
        t.append(t_current)
        t_current = t_current + h
    return np.array(t), y


def runge_kutta_5(func, t_0, t_f, y_0, h):
    """ Approximates the solution to the differential equation given by dy/dt=func(t, y) over the interval [t0, tf]
    with initial conditions (t0, y0) using Runge kutta's fifth order method with step-size h
    Args:
        func ((float, float)->float): This is the derivative of the dependent variable with respect to the single independent
        variable
        t_0 (float): the value of the single independent variable at the initial condition
        t_f (float): the final desired value
        y_0 (float): the value of the dependent variable at the initial condition
        h (float): the step-size of euler's method (use negative to go backwards in t)
    Returns:
     Numpy arrays (with floats) that contain the solution to dy/dt=func(t, y)
     The format is (T, Y) where Y[i] is the image of the t value given by T[i]
    """
    t = [t_0]
    y = [y_0]
    t_current = t_0 + h
    while (t_current <= t_f):
        k1 = func(t[-1], y[-1])
        k2 = func(t[-1] + h / 4.0, y[-1] + k1 * h / 4.0)
        k3 = func(t[-1] + h / 4.0, y[-1] + k1 * h / 8.0 + k2 * h / 8.0)
        k4 = func(t[-1] + h / 2.0, y[-1] - k2 * h / 2.0 + k3 * h)
        k5 = func(t[-1] + 3.0 * h / 4.0, y[-1] + 3.0 / 16.0 * k1 * h + 9.0 / 16.0 * k4 * h)
        k6 = func(t[-1] + h, y[
            -1] - 3.0 / 7.0 * k1 * h + 2.0 / 7.0 * k2 * h + 12.0 / 7.0 * k3 * h - 12.0 / 7.0 * k4 * h + 8.0 / 7.0 * k5 * h)
        y.append(y[-1] + h / 90.0 * (7.0 * k1 + 32.0 * k3 + 12.0 * k4 + 32.0 * k5 + 7.0 * k6))
        t.append(t_current)
        t_current = t_current + h
    return np.array(t), np.array(y)
