# Numerical Methods 
Provides Methods for Numerical Methods. Includes tools for root-finding, integration, Ordinary differential equations (initial value problems for single equations or systems of equations), optimization, linear algebra, differentiation, and curve fitting. This code was written back in 2018 as a side project while I was taking a class on Numerical Methods.

# Dependencies
* The example application requires Matplotlib
* Numpy (Mostly for Numpy arrays, and for simple arithmetic such as np.log(), np.abs(), np.sqrt(), etc -- This library provides everything else)

# Examples
Please see and run examples.py. Make sure that you can see the matplotlib output. 

# Usage
Every function has a docstring explaining what it does, returns, what its parameters mean, and how to use it. The examples
inside examples.py should also be very helpful. 

Every function has been tested to work with appropriate accuracy as long as the assumptions are met (For example: most methods require function or data to be real valued, many methods require functions to be continous, etc).

# Applied Example

The following set of first order differential equations:
```
dx/dt = v

dv/dt = g - cd / m * v ** 2
```

The analytical solution is:
```
x(t) = m * ln(cosh(sqrt(g * cd / m) * t)) / cd

v(t) = sqrt(gm/cd) tanh(sqrt(gcd/m)t)
```

Let g = 9.81, cd = 0.25, m = 68.1, step-size = 0.5
And initial conditions at time t = 0 are  x(0) = v(0) = 0

<br>

``` python
    # Exact result
    def solX(t):
        g = 9.81
        cd = 0.25
        m = 68.1
        return m * np.log(np.cosh(np.sqrt(g * cd / m) * t)) / cd

    def solV(t):
        g = 9.81
        cd = 0.25
        m = 68.1
        return np.sqrt(g * m / cd) * np.tanh(np.sqrt(g * cd / m) * t)
    
    # Define functions in the system. f1(t, x, v) = dx/dt and f2(t, x, v) = dv/dt
    # Since there are two dependent variables in our system, y[0] represents x and y[1] represents v
    def f1(t, y):
        return y[1]

    def f2(t, y):
        g = 9.81
        cd = 0.25
        m = 68.1
        return g - cd / m * y[1] ** 2

    # Make a list of the equations
    functions = [f1, f2]
    
    # initial conditions, y0[0] corresponds to x(0) and y0[1] corresponds to v(0)
    y0 = [0.0, 0.0]
    
    # call my library and use system_ode_runge_kutta_4 
    t, y = ordinary_diff_eq.system_ode_runge_kutta_4(functions, 0.0, 10.0, y0, h=0.5)
    
    # the first column corresponds to x, the second column to v
    x = y[0, :]
    v = y[1, :]
    
    # Plot analytical x(t) vs estimated x(t) and analytical v(t) vs estimated v(t)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, x, label="Estimated x (RK 4)")
    plt.plot(t, solX(t), label="Analytical x")
    plt.legend()
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(t, v, label="Estimated v (RK 4)")
    plt.plot(t, solV(t), label="Analytical v")
    plt.grid()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.legend()
    plt.show()
```

This is a graph of the analytical solution and the estimated solution (Units are ommited since this example is used only for 
illustration purposes). Notice that the blue and orange curves overlap for both x(t) and v(t), indicating highly accurate results. 

<a href="/images/RK4.png"> Click Here for a High Resolution Version of the Image </a>
<img src="images/RK4.png?raw=true"/>
