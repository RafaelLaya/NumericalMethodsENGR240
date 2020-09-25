import roots
import quadrature
import ordinary_diff_eq
import optimization
import linear_algebra
import differentiation
import curve_fitting
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

"""
            These examples will be helpful to show some of the classical ways that these functions can be used 
            Most of these are complex enough to boost the user's confidence that the results are correct. while being 
            simple enough that can be checked by hand.
"""


def examples_roots():
    print("========== ROOTS ==========")
    print("The function x*(x^3-x^2+x-1)*exp(x) has a root x=1")

    def f(x):
        return x * (x ** 3 - x ** 2 + x - 1) * np.exp(x)

    print("Some estimated values are: ")
    print(roots.brents_method(f, 0.5, 4))
    print(roots.incremental_search(f, 0.5, 4, h=0.01)[0])
    print(roots.incremental_with_bisection(f, 0.5, 4)[0])
    print(roots.bisection(f, 0.5, 4))
    print(roots.false_position(f, 0.5, 4))
    print(roots.secant_method(f, 3))

    print("\nAnother root is at x=0")
    print("Some estimates:")
    print(roots.brents_method(f, -3, 0.5))
    print(roots.incremental_search(f, -3, 0.5, h=0.1)[0])
    print(roots.incremental_with_bisection(f, -3, 0.5)[0])
    print(roots.bisection(f, -3, 0.5))
    print(roots.false_position(f, -3, 0.5))
    print(roots.secant_method(f, -0.5))

    print("\nIncremental search can try to find multiple roots")
    print("This can be useful when you don't know how to bound the roots")
    print("I am gonna use a small h=0.5 It is important that two roots are at least more than h apart")
    print(roots.incremental_search(f, -5.0, 5.0, h=0.5))
    print(roots.incremental_with_bisection(f, -5.0, 5.0, h=0.5))
    print("\nYou can use small h (the biggest spacing between roots will be h) to get more accurate results")
    print("My favourite way is to use incremental search to find good starting points and then end with newton's method")

    vals = roots.incremental_search(f, -5.0, 5.0, h=0.9)
    for val in vals:
        print(roots.secant_method(f, val))

    print()
    print("Let's now apply the multidimensional version of Newton's method")
    print("g1(x, y, z) = 3x - cos(y*z) - 3 / 2")
    print("g2(x, y, z) = 4x^2 - 625y^2 + 2z - 1")
    print("g3(x, y, z) = 20 z + e^(-xy) + 9")

    def g_1(x):
        return 3 * x[0] - np.cos(x[1] * x[2]) - 3.0 / 2.0

    def g_2(x):
        return 4 * x[0] ** 2 - 625 * x[1] ** 2 + 2 * x[2] - 1

    def g_3(x):
        return 20 * x[2] + np.exp(- x[0] * x[1]) + 9

    print("If the initial condition is x=1.0, y=1.0, z=1.0")
    print("This will be the estimated root: ")
    x_0 = [1.0, 1.0, 1.0]
    result = roots.newton_multi_var([g_1, g_2, g_3], x_0)
    print("x = ", result[0][0])
    print("y = ", result[1][0])
    print("z = ", result[2][0])
    print("\nLet's use the roots we discovered to confirm that the original functions are indeed zero at the roots: ")
    print("g1 = ", g_1(result))
    print("g2 = ", g_2(result))
    print("g3 = ", g_3(result))


def examples_optimization():
    print("========== OPTIMIZATION ==========")
    print("h(x) = -x^3 - x^2 + x - 1 has a maximum when x=1/3 and a minimum when x=-1")

    def h(x):
        return (- x ** 3 - x ** 2 + x - 1)

    print("Let's calculate some estimates. First, use golden section:")
    print("x1 = {}".format(optimization.golden_section_max_search(h, 0.0, 2.0)))
    print("x2 = {}".format(optimization.golden_section_min_search(h, -2.0, 0.0)))

    print("\nNow parabolic interpolation:")
    print("x1 = {}".format(optimization.parabolic_interp_max_search(h, 0.0, 1.0, 2.0)))
    print("x2 = {}".format(optimization.parabolic_interp_min_search(h, -2.0, -0.5, 0.0)))

    print("\nNow brent's method:")
    print("x1 = {}".format(optimization.brent_max_search(h, 0.0, 0.1, 2.0)))
    print("x2 = {}".format(optimization.brent_min_search(h, -2.0, -0.5, 0.0)))

    print("\nLet's try some multivariable optimization")
    print("f(x) = e^(-1/3 * x^3 + x - y ^ 2) has a local minimum at x = 1, y = 0")
    print("Some estimates:")

    def f(x):
        return np.exp(-1.0 / 3.0 * x[0] ** 3 + x[0] - x[1] ** 2)

    print("[x, y] = {}".format(optimization.gradient_ascent(f, [0.5, 0.5])))
    print("[x, y] = {}".format(optimization.gradient_ascent(f, np.array([0.5, 0.5]))))
    print("[x, y] = {}".format(optimization.gradient_ascent(f, np.array([0.5, 0.5]).reshape(2, 1))))
    print("[x, y] = {}".format(optimization.gradient_ascent(f, np.array([0.5, 0.5]).reshape(2, ))))

    print()
    print("h(x) = -e^(-1/3 * x^3 + x - y ^ 2) has a local maximum at x = 1, y = 0")
    print("Some estimates:")

    def h(x):
        return - f(x)

    print("[x, y] = {}".format(optimization.gradient_descend(h, [0.5, 0.5])))
    print("[x, y] = {}".format(optimization.gradient_descend(h, np.array([0.5, 0.5]))))
    print("[x, y] = {}".format(optimization.gradient_descend(h, np.array([0.5, 0.5]).reshape(2, 1))))
    print("[x, y] = {}".format(optimization.gradient_descend(h, np.array([0.5, 0.5]).reshape(2, ))))


def examples_quadrature():
    print("========== QUADRATURE ==========")
    print("f(x) = x^3 - x^2 + x - 1 + cos(x)e^x")
    print("The integral of f over [-5, 5] is:")
    print("-280/3 + cos(5)sinh(5) + sin(5)cosh(5)")
    print("or about -143.446")

    def f(x):
        return x ** 3 - x ** 2 + x - 1 + np.exp(x) * np.cos(x)

    t = np.linspace(-5.0, 5.0, 101)
    y = f(t)
    print("Cumulative means that it calculates intermediate results")
    print("This is useful if you'd like to calculate the integral of the same function over similar intervals")
    print("\nSome estimates using composite, cumulative, and adaptive quadrature:")
    print("Trapezoidal Composite: {}".format(quadrature.trapezoidal_composite_F(f, -5.0, 5.0, 100)))
    print("Trapezoidal Composite (different syntax): {}".format(quadrature.trapezoidal_composite(t, y)))
    print("Trapezoidal Cumulative: {}".format(quadrature.trapezoidal_cumulative(t, y)[-5:]))
    print("Trapezoidal Cumulative (different syntax): {}".format(quadrature.trapezoidal_cumulative_F(f, -5.0, 5.0, 100)[-5:]))

    result = quadrature.trapezoidal_cumulative(t, y)
    print("Using the Trapezoidal Cumulative method, we can find similar integrals")
    print("Example: The Integral from x =", t[(int)(len(t) / 2.0)])
    print("To x =", t[-1], "is:")
    print(result[-1] - result[(int)(len(t) / 2.0)])

    print("\nMore estimates (from -5 to 5)")
    print()
    print("Simpson's Method: {}".format(quadrature.simpson(t, y)))
    print("Simpson's Method (different syntax): {}".format(quadrature.simpson_F(f, -5.0, 5.0, 100)))
    print("Simpson's 1/3 Method: {}".format(quadrature.simpson_one_third_composite_F(f, -5.0, 5.0, 100)))
    print("Simpson's 1/3 Composite Method: {}".format(quadrature.simpson_one_third_composite(t, y)))

    print()
    print("Romberg's Method: {}".format(quadrature.romberg(f, -5.0, 5.0, 100)))
    print("Adaptive quadrature: {}".format(quadrature.adaptive(f, -5.0, 5.0)))

    print()
    print("A composite method partitions an interval and uses single-rule applications on each small interval.")
    print("Single-rule applications obviously offer a big error on big intervals. ")
    print("The following are single-rule applications (non-composite) applications:")
    print("Simpson's 1/3 Rule: {}".format(quadrature.simpson_one_third_rule(-5.0, 0.0, 5.0, f(-5.0), f(0.0), f(5.0))))
    print("Simpson's 3/8 Rule: {}".format(quadrature.simpson_three_eight_rule(-5.0, -2.0, 2.0, 5.0, f(-5.0), f(-2.0), f(2.), f(5.0))))
    print("Boole's 5th degree Rule: {}".format(quadrature.boole_rule_5(-5.0, -2.5, 0.0, 2.5, 5.0, f(-5.0), f(-2.5), f(0.0), f(2.5), f(5.0))))
    print("Boole's 6th degree Rule: {}".format(quadrature.boole_rule_6(-5.0, -3.0, -0.5, 0.5, 3.0, 5.0, f(-5.0), f(-3.0), f(-0.5), f(0.5), f(3.0), f(5.0))))
    print("Trapezoidal Rule: {}".format(quadrature.trapezoidal_rule(-5.0, 5.0, f(-5.0), f(5.0))))
    print("Gauss-Legendre 2nd degree Rule: {}".format(quadrature.gauss_leg_2(f, -5.0, 5.0)))
    print("Gauss-Legendre 3rd degree Rule: {}".format(quadrature.gauss_leg_3(f, -5.0, 5.0)))
    print("Gauss-Legendre 6th degree Rule: {}".format(quadrature.gauss_leg_6(f, -5.0, 5.0)))

    # open quadrature
    print("\n Let's try Open Quadrature methods")
    print("The function g(x) = 1.0 / sqrt(3-x)")
    print("Its integral over [0, 3] results in 2sqrt(3) or about 3.4641")
    print("However, note that the integral is improper")
    print("Open quadrature methods are required")
    print("Some estimates below:")

    def g(x):
        return 1.0 / np.sqrt(3.0 - x)

    print("Adaptive Open Quadrature: {}".format(quadrature.open_adaptive(g, 0.0, 3.0, 1000)))
    print("Open Quadrature: {}".format(quadrature.open_quad(g, 0.0, 3.0, 10000)))
    print("Open Quadrature with Romberg's Method: {}".format(quadrature.open_romberg(g, 0.0, 3.0, 1000)))

    print("\nSingle rule applications of Open Quadrature:")
    print("Midpoint Method with 2 Points in Open Quadrature: {}".format(quadrature.open_midpoint_2(g, 0.0, 3.0)))
    print("Midpoint Method with 3 Points in Open Quadrature: {}".format(quadrature.open_midpoint_3(g, 0.0, 3.0)))
    print("Midpoint Method with 4 Points in Open Quadrature: {}".format(quadrature.open_midpoint_4(g, 0.0, 3.0)))
    print("Midpoint Method with 5 Points in Open Quadrature: {}".format(quadrature.open_midpoint_5(g, 0.0, 3.0)))
    print("Midpoint Method with 6 Points in Open Quadrature: {}".format(quadrature.open_midpoint_6(g, 0.0, 3.0)))

    print()
    print("let's do double quadrature")
    print("Also known as double integration")
    print("p(x, y) = xy - 3xy^2")
    print("Integration of p over the rectangle [0, 2] x [1, 2] results in -11")
    print("Some estimates: ")

    def p(x, y):
        return x * y - 3 * x * y ** 2

    x = np.linspace(0.0, 2.0, 100)
    y = np.linspace(1.0, 2.0, 100)
    Z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i][j] = p(x[i], y[j])
    print("\nUsing Simpson's method of double quadrature: {}".format(quadrature.simpson_double(x, y, Z)))
    print("Using the Trapezoidal Rule of double quadrature: {}".format(quadrature.trapezoidal_double(x, y, Z)))

    x = np.linspace(0.0, 2.0, 100)
    y = np.linspace(1.0, 2.0, 100)
    X, Y = np.meshgrid(x, y)
    Z = p(X, Y)
    print("\nUsing Simpson's method of double quadrature: {}".format(quadrature.simpson_double(x, y, Z)))
    print("Using the Trapezoidal Rule of double quadrature: {}".format(quadrature.trapezoidal_double(x, y, Z)))


def examples_curve_fitting():
    print("========== CURVE FITTING ==========")
    print("Given the following ordered data")
    x = [1.0, 3.0, 4.0, 2.0, 5.0, 5.5, 6.0, 7.0, 1.5, 1.6]
    y = [2.0, 2.5, 4.0, 4.5, 4.0, 6.0, 8.7, 10.0, 0.4, 0.8]
    x, y = (list(t) for t in zip(*sorted(zip(x, y))))
    print(x)
    print(y)
    print("\nWe can look at the linear regression: ")
    print("R^2 =", curve_fitting.r_coefficient(x, y) ** 2)
    slope, intercept = curve_fitting.linear_reg(x, y)
    print("Slope =", slope)
    print("Intercept =", intercept)

    print()
    print("We can also use linear regression while forcing the line to pass through the origin: ")
    # x = x.reshape(len(x), )
    # y = y.reshape(len(x), )
    slope, intercept = curve_fitting.linear_reg_through_origin(x, y)
    print("Slope =", slope)
    print("Intercept =", intercept)

    print()
    print("Now let's evaluate a polynomial")
    print("p(x) = 5x + 2")
    print("If x = 3, then p(x) = ", end="")
    slope = 5.0
    intercept = 2.0
    coeff = [5.0, 2.0]
    print(curve_fitting.poly_eval(coeff, 3.0))  # = 5x + 2 at x=3

    print("It is also possible to evaluate a polynomial at different points using lists. ")
    print("For example, evaluate at x=1, x=2 and x=3")
    t = [1.0, 2.0, 3.0]
    print(curve_fitting.poly_eval(coeff, t))

    print()
    print("Given the following ordered data: ")
    x = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    y = [25.0, 70.0, 380.0, 550.0, 610.0, 1220.0, 830.0, 1450.0]
    x, y = (list(t) for t in zip(*sorted(zip(x, y))))
    print(x)
    print(y)
    print("We can fit this to an exponential model y = alpha * e ^(beta * x)")
    alpha, beta = curve_fitting.fit_power_model(x, y)
    print("Alpha =", alpha)
    print("Beta =", beta)

    t = np.linspace(min(np.log(x)), max(np.log(x)), 100)
    plt.plot(np.log(x), np.log(y), "r.")
    plt.plot(t, np.log(alpha) + beta * t)
    plt.show()

    print("\nMore examples of doing the same thing, using slightly different syntax (see code): ")
    x = np.array(x)
    y = np.array(y)
    x, y = (list(t) for t in zip(*sorted(zip(x, y))))
    alpha, beta = curve_fitting.fit_power_model(x, y)
    print("Alpha =", alpha)
    print("Beta =", beta)

    func = curve_fitting.fit_power_model_F(x, y)
    print("\nAlpha =", func(0))
    print("Beta =", np.log(func(5.0) / func(0)) / 5.0)

    print("\nNow we will run some examples with splines: ")
    problem12()
    problem9()
    problem14a()

    x = np.array([2.0, 9.0])
    y = np.array([1.0, 6.0])
    A = np.array([[60.0, 55],
                  [57.5, 70]])
    print("\nWith the following data: ")
    print("x = ", x)
    print("y = ", y)
    print("Let A[i][j] be the image of this function = \n", A)
    print("We can interpolate the value when x = 5.25, y = 4.8")

    print()
    print("Bilinear interpolation with Newton's Polynomial: {}".format(curve_fitting.bilinear_interp_newt(x, y, A)(5.25, 4.8)))
    print("Bilinear interpolation with Lagrange's Polynomial: {}".format(curve_fitting.bilinear_interp_lagrange(x, y, A)(5.25, 4.8)))
    print("Bilinear interpolation with four points: {}".format(curve_fitting.bilinear_interp_four_pts(x, y, A)(5.25, 4.8)))
    print("Bilinear interpolation with Cubic Splines: {}".format(curve_fitting.bilinear_interp_cubic_splines(x, y, A)(5.25, 4.8)))
    print("Bilinear interpolation with Linear Splines: {}".format(curve_fitting.bilinear_interp_linear_splines(x, y, A)(5.25, 4.8)))
    print("Bilinear interpolation: {}".format(curve_fitting.bilinear_interp(x, y, A)(5.25, 4.8)))

def problem14a():
    print("========== EXAMPLE ==========")
    print("And now, some bilinear interpolation: ")
    print("T(z, w) = 2 + z - w + 2 z^2 + 2 z * w + w^2")

    def T(z, w):
        return 2 + z - w + 2 * z ** 2 + 2 * z * w + w ** 2

    x = np.linspace(-2, 0, 9)
    y = np.linspace(0, 3, 9)
    X, Y = np.meshgrid(x, y)
    Z = T(X, Y)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z)
    plt.show()

    print("Evaluated at z = -1.63 and w = 1.627: ")
    print("T(z,w)={}".format(T(-1.63, 1.627)))
    print("Some estimates: ")
    print("Bilinear interpolation: {}".format(curve_fitting.bilinear_interp(x, y, Z)(-1.63, 1.627)))
    print("Bilinear interpolation with Lagrange's Polynomial: {}".format(curve_fitting.bilinear_interp_lagrange(x, y, Z)(-1.63, 1.627)))
    print("Bilinear interpolation with Newton's Polynomial: {}".format(curve_fitting.bilinear_interp_newt(x, y, Z)(-1.63, 1.627)))
    print("Bilinear interpolation with Cubic Splines: {}".format(curve_fitting.bilinear_interp_cubic_splines(x, y, Z)(-1.63, 1.627)))
    print("Bilinear interpolation with Linear Splines: {}".format(curve_fitting.bilinear_interp_linear_splines(x, y, Z)(-1.63, 1.627)))

def problem12():
    def f(t):
        return (np.sin(t)) ** 2

    T = np.linspace(0, 2 * np.pi, 2 * 4 * 20)
    t = np.linspace(0, 2 * np.pi, 8)

    problem12a(f, T, t)
    problem12b(f, T, t)


def problem12a(f, T, t):
    print("========== EXAMPLE ==========")
    print("Let's use Natural Cubic Splines")
    cubic = curve_fitting.cubic_splines(t, f(t))
    plt.plot(t, f(t), "r.", markersize=20, label="data")
    plt.plot(T, cubic(T), label="Cubic Natural Spline")
    plt.plot(T, np.abs(cubic(T) - f(T)), label="error")
    plt.legend()
    plt.grid()
    plt.show()

    print("Repeated, but using lists:")
    t = t.tolist()
    T = T.tolist()
    cubic = curve_fitting.cubic_splines(t, f(t))
    plt.plot(t, f(t), "r.", markersize=20, label="data")
    plt.plot(T, cubic(T), label="Cubic Natural Spline")
    plt.plot(T, np.abs(cubic(T) - f(T)), label="error")
    plt.legend()
    plt.grid()
    plt.show()


def problem12b(f, T, t):
    print("========== EXAMPLE ==========")
    print("Let's use Cubic Clamped Splines")
    def dfdt(z):
        h = 0.0001
        return (f(z + h) - f(z - h)) / (2.0 * h)

    clamped = curve_fitting.clamped_splines(t, f(t), dfdt(t[0]), dfdt(t[-1]))
    plt.plot(t, f(t), "r.", markersize=20, label="data")
    plt.plot(T, clamped(T), label="Cubic Clamped Splines")
    plt.plot(T, np.abs(clamped(T) - f(T)), label="error")
    plt.legend()
    plt.grid()
    plt.show()

    print("Again, but using lists:")
    t = t.tolist()
    T = T.tolist()
    clamped = curve_fitting.clamped_splines(t, f(t), dfdt(t[0]), dfdt(t[-1]))
    plt.plot(t, f(t), "r.", markersize=20, label="data")
    plt.plot(T, clamped(T), label="Cubic Clamped Splines")
    plt.plot(T, np.abs(clamped(T) - f(T)), label="error")
    plt.legend()
    plt.grid()
    plt.show()


def problem9():
    print("========== EXAMPLE ==========")
    print("Linear Splines vs Newton Polynomial vs Cubic Splines vs Lagrange Polynomial")
    T = np.array([0.0, 8, 16, 24, 32, 40])
    o = np.array([14.621, 11.843, 9.870, 8.418, 7.305, 6.413])
    X = np.linspace(min(T), max(T), int((max(T) - min(T)) * 20 + 1))

    linear_spl = curve_fitting.linear_splines(T, o)
    fifth_order_newt = curve_fitting.newton_interp_poly(T, o)
    cubic_spl = curve_fitting.cubic_splines(T, o)
    fifth_order_lagr = curve_fitting.lagrange_poly(T, o)

    plt.plot(X, linear_spl(X), label="linear Splines")
    plt.plot(T, o, "r.", markersize=20)
    plt.plot(X, fifth_order_newt(X), label="Newton Polynomial")
    plt.plot(X, cubic_spl(X), label="Cubic Spline")
    plt.plot(X, fifth_order_lagr(X), label="Lagrange Polynomial")
    plt.legend()
    plt.grid()
    plt.show()


def examples_linear_algebra():
    print("========== LINEAR ALGEBRA ==========")
    print("Solving Ax=b where ")
    A = np.array([[6.0, 5.0, 3.2],
                  [5.5, -5.2, 3.4],
                  [5.2, -10.9, -3.3]])
    B = np.array([[5.0, 6., -2],
                  [-3, -4, 2],
                  [5.3, -12, 14]])
    b = np.array([[5.0],
                  [7.0],
                  [8.0]])
    print("A = \n{}\n".format(A))
    print("b = \n{}\n".format(b))
    print("Some estimates: ")
    print("Cramer's Method: \n{}".format(linear_algebra.cramers(A, b)))
    print("Cramer's Method (numpy for comparison): \n{}".format(linear_algebra.cramers_np(A, b)))
    print("Gauss Elimination: \n{}".format(linear_algebra.gauss_elimination(A, b)[0]))
    P, L, U = linear_algebra.pluDecomp(A)
    print("Using PLU Decomposition: \n{}".format(linear_algebra.pluSolver(P, L, U, b)))

    print("We can verify that Ax - b = 0")
    print(linear_algebra.matrix_mult(A, linear_algebra.gauss_elimination(A, b)[0]) - b)

    print("\nLet B = \n{}\n".format(B))
    print("\nA times B is: \n{}\n".format(linear_algebra.matrix_mult(A, B)))

    print("\ndet(A): \n{}\n".format(linear_algebra.determinant(A)))
    print("\ndet(A) Using Gauss Elimination: \n{}\n".format(linear_algebra.determinant_gauss_elim(A)))

    A = np.array([[4.0, 12.0, -16.0],
                  [12.0, 37.0, -43.0],
                  [-16.0, -43.0, 98.0]])
    print("\nLet A = \n", A)
    print("Its cholesky decomposition is: ")
    U = linear_algebra.cholesky_dec(A)
    print(U)
    print("We can verify U^T * U - A = 0")
    print(linear_algebra.matrix_mult(np.transpose(U), U) - A)
    print()
    print("The inverse of A: ")
    print(linear_algebra.inverse_elim(A))
    print(linear_algebra.inverse_lu(A))
    P, L, U = linear_algebra.pluDecomp(A)
    print(linear_algebra.inverse_given_lu(P, L, U))

    print("\n We can verify that A times its inverse is the identity: ")
    Ainv = linear_algebra.inverse_elim(A)
    print(linear_algebra.matrix_mult(A, Ainv))
    print()
    print(linear_algebra.matrix_mult(Ainv, A))
    print()
    print("\nLet's look at the condition number of A using different norms: ")
    print("Frobenius Norm: {}".format(linear_algebra.condition(A, "fro")))
    print("Column Norm: {}".format(linear_algebra.condition(A, "col")))
    print("Row Norm: {}".format(linear_algebra.condition(A, "row")))
    print("Eigenvalues Norm: {}".format(linear_algebra.condition(A, "eig")))
    print()

    A = np.array([[3, -2, 1],
                  [1, -3, 2],
                  [-1, 2, 4]])
    print("\nLet A = \n{}\n".format(A))
    print("b = \n{}\n".format(b))
    print("We can also solve using gauss-seidel: ")
    print(linear_algebra.gauss_seidel(A, b))
    print("Verification that Ax - b = 0")
    print(linear_algebra.matrix_mult(A, linear_algebra.gauss_seidel(A, b)) - b)

    print()
    print("Distinct real eigenvalues and eigenvectors")
    print("Let A = \n", A)
    val, vec = linear_algebra.eig_power_method(A)
    print("Eigenvalue: \n", val)
    print("Eigenvector: \n", vec)
    print("Verification A(vector) - (value)(vector) = 0")
    print(A @ vec - val[0] * vec)

    print()
    print("Using another method (to find all eigenvalues): ")
    vals = linear_algebra.eigenvals_M(A)
    print("eigenvalues: \n", vals)
    print("Verification det(A - (value) * Identity) = 0:")
    for val in vals:
        print("For value {}: {}".format(val, linear_algebra.determinant_gauss_elim(A - val * np.eye(3))))

    print("\nTridiagonal solver")
    A = np.array([[1, 4, 0, 0],
                  [3, 4, 1, 0],
                  [0, 2, 3, 4],
                  [0, 0, 1, 3]])
    print("Let A = \n{}\n".format(A))
    b = np.array([[4],
                  [3],
                  [6],
                  [8]])
    print("Let b = \n{}\n".format(b))
    
    print("The solution to Ax = b is: ")
    print(linear_algebra.tridiagonal_solver(A, b))
    print("Check that Ax - b = 0")
    print(linear_algebra.matrix_mult(A, linear_algebra.tridiagonal_solver(A, b)) - b)

    print()
    print("Let A = ")
    A = np.array([[12.0, -51.0, 4.0],
                  [6.0, 167.0, -68.0],
                  [-4.0, 24.0, -41.0]])
    print(A)
    print("\nIts QR decomposition is: ")
    Q, R = linear_algebra.QR_decomposition(A)
    print("Q = \n{}\n".format(Q))
    print("R = \n{}\n".format(R))
    print("Verify that Q @ R - A = 0")
    print(linear_algebra.matrix_mult(Q, R) - A)

    print()
    print("Let A = ")

    A = np.array([[1.0, 3, 0, 0],
                  [3, 2, 1, 0.0],
                  [0.0, 1, 3, 4],
                  [0.0, 0, 4, 1]])
    print(A)
    print("The QR algorithm finds eigenvectors and eigenvalues")
    T, U = linear_algebra.eigQR(A)
    print("Values: \n{}\n".format(T))
    print("Vectors (column i corresponds to value i above): \n", U)
    print("\nVerification that A(vector) - value(vector) = 0")
    for i in range(np.shape(A)[0]):
        print("For value {} and vector \n{}\nA*vector-value*vector should be zero:".format(T[i], U[:,i]))
        print(A @ U[:, i].reshape(np.shape(A)[0], 1) - T[i] * U[:, i].reshape(np.shape(A)[0], 1))
        print("\n\n\n")

    A = np.array([[1.0, 3, 4],
                  [3.0, 1, 2],
                  [4.0, 2, 1]])
    print("\nLet A =\n{}\n".format(A))
    T, U = linear_algebra.eigQR(A)

    print("Values: \n", T)
    print("Vectors (as columns): \n", U)
    print("\nVerification that A(vector) - value(vector) = 0")
    
    for i in range(np.shape(A)[0]):
        print("For value {} and vector \n{}\nA*vector-value*vector should be zero:".format(T[i], U[:,i]))
        print(A @ U[:, i].reshape(np.shape(A)[0], 1) - T[i] * U[:, i].reshape(np.shape(A)[0], 1))
        print("\n\n\n")


def examples_differentiation():
    print("========== DIFFERENTIATION ==========")
    print("f(x) = x^3 - x^2 + x - 1 + e^(x) * cos(x)")
    print("df/dx at x=5 is 66 - e^5 (cos(5) - sin(5))")
    print("Or about 250.416182")
    print("Some estimates: ")

    def f(x):
        return x ** 3 - x ** 2 + x - 1 + np.exp(x) * np.cos(x)

    print(differentiation.first_derivative_backwards(f, 5.0))
    print(differentiation.first_derivative_centered(f, 5.0))
    print(differentiation.first_derivative_forward(f, 5.0))

    print()
    print("The second derivative at x = 5 is 28 - 2e^5 sin(5)")
    print("Or about 312.6339619")
    print(differentiation.second_derivative_backwards(f, 5.0))
    print(differentiation.second_derivative_centered(f, 5.0))
    print(differentiation.second_derivative_forward(f, 5.0))

    print()
    print("Third derivative: -2(-3 + e^5 sin(5) + e^5 cos(5))")
    print("About 206.4355598")
    print(differentiation.third_derivative_backward(f, 5.0))
    print(differentiation.third_derivative_centered(f, 5.0))
    print(differentiation.third_derivative_forward(f, 5.0))

    print()
    print("Fourth derivative: -4e^5 cos(5)")
    print("About -168.3968049")
    print(differentiation.fourth_derivative_backward(f, 5.0))
    print(differentiation.fourth_derivative_centered(f, 5.0))
    print(differentiation.fourth_derivative_forward(f, 5.0))

    print("\nNow using the API to get the nth derivative:")
    for i in range(1, 5, 1):
        print("Order: ", i)
        print(differentiation.deriv_n(f, 5.0, i, h=0.0008))

    print("\nAlso works with lists: (evaluates derivative at each value from the vector)")
    t = np.linspace(0, 5, 6)
    print()
    for i in range(1, 5, 1):
        print("Derivative of order {} on {} is \n{}\n".format(i, t, differentiation.deriv_n(f, t, i)))

    print("\nNow more first order derivatives through various methods: ")
    print("\nRichardson extrapolation: ")
    print(differentiation.richardson_extrapolation(f, 5.0))
    print("\nAlso works with lists: (evaluates derivative at each value from the vector)")
    print(differentiation.richardson_extrapolation(f, t))
    print("\nThe same goes for romberg's method: ")
    print(differentiation.derivative_romberg(f, 5.0))
    print(differentiation.derivative_romberg(f, t))

    print()
    print("\nUsing lagrange polynomial: ")
    x = np.linspace(0.0, 10.0, 100)
    y = f(x)
    print(differentiation.derivative_lagrange_poly(x, y, 5.0))
    print(differentiation.derivative_lagrange_poly(x, y, t))

    print("\nWe can also differentiate multivariable functions")
    print("Let multi(x, y) = x ^ 2 * y")

    def multi(x):
        return x[0] ** 2 * x[1]

    print("We can take partial derivatives")
    print("x = 3.0, y = 2.0")
    print("\nPartial respect to x: ")
    print(differentiation.partialDerivative(multi, [3.0, 2.0], 0))
    print("\nPartial respect to y: ")
    print(differentiation.partialDerivative(multi, [3.0, 2.0], 1))

    print("\nAlso gradients: ")
    print(differentiation.gradient(multi, [3.0, 2.0]))
    print(differentiation.gradient_f(multi)([3.0, 2.0]))

def examples_ODE():
    print("========== ORDINARY DIFFERENTIAL EQUATIONS ==========")
    print("Trying to solve: dy/dt + 2y = 2 - e^(-4y) with y(0) = 1")
    print("This implies dy/dt = f(t, y) = 2 - e^(-4t) - 2 y")
    print("The solution is 1.0 + 0.5 e^(-4t) - 0.5 e ^(-2t)")

    def f(t, y):
        return 2 - np.exp(-4 * t) - 2 * y

    t, y = ordinary_diff_eq.euler(f, 0, 10.0, 1.0, h=0.1)
    plt.plot(t, y, label="Euler's method")
    plt.plot(t, 1.0 + 0.5 * np.exp(-4 * t) - 0.5 * np.exp(-2 * t), label="Analytical Solution")
    plt.legend()
    plt.show()

    t, y = ordinary_diff_eq.heun_iterative(f, 0, 10.0, 1.0, h=0.1)
    plt.plot(t, y, label="Heun's method (iterative)")
    plt.plot(t, 1.0 + 0.5 * np.exp(-4 * t) - 0.5 * np.exp(-2 * t), label="Analytical Solution")
    plt.legend()
    plt.show()

    t, y = ordinary_diff_eq.heun_not_iterative(f, 0, 10.0, 1.0, h=0.1)
    plt.plot(t, y, label="Heun's method (not iterative)")
    plt.plot(t, 1.0 + 0.5 * np.exp(-4 * t) - 0.5 * np.exp(-2 * t), label="Analytical Solution")
    plt.legend()
    plt.show()

    t, y = ordinary_diff_eq.midpoint_method(f, 0, 10.0, 1.0, h=0.1)
    plt.plot(t, y, label="Midpoint Method")
    plt.plot(t, 1.0 + 0.5 * np.exp(-4 * t) - 0.5 * np.exp(-2 * t), label="Analytical Solution")
    plt.legend()
    plt.show()

    t, y = ordinary_diff_eq.ralston_method(f, 0, 10.0, 1.0, h=0.1)
    plt.plot(t, y, label="Ralstons method")
    plt.plot(t, 1.0 + 0.5 * np.exp(-4 * t) - 0.5 * np.exp(-2 * t), label="Analytical Solution")
    plt.legend()
    plt.show()

    t, y = ordinary_diff_eq.runge_kutta_4(f, 0, 10.0, 1.0, h=0.1)
    plt.plot(t, y, label="Runge Kutta 4th order")
    plt.plot(t, 1.0 + 0.5 * np.exp(-4 * t) - 0.5 * np.exp(-2 * t), label="Analytical Solution")
    plt.legend()
    plt.show()

    t, y = ordinary_diff_eq.runge_kutta_5(f, 0, 10.0, 1.0, h=0.1)
    plt.plot(t, y, label="Runge Kutta 5th Order")
    plt.plot(t, 1.0 + 0.5 * np.exp(-4 * t) - 0.5 * np.exp(-2 * t), label="Analytical Solution")
    plt.legend()
    plt.show()

    print("\n Let's do system of first order ODEs now: ")
    ODE_system_example(ordinary_diff_eq.system_ode_euler, "Estimated x (euler's method", "Estimated v (euler's method)")
    ODE_system_example(ordinary_diff_eq.system_ode_runge_kutta_4, "Estimated x (RK 4)", "Estimated v (RK 4)")


def ODE_system_example(function, lab1, lab2):
    print("========== EXAMPLE ==========")
    print("The following set of first order differential equations:")
    print("dx/dt = v")
    print("dv/dt = g - cd / m * v ** 2")
    print("The solution is x(t) = m * ln(cosh(sqrt(g * cd / m) * t)) / cd")
    print("v(t) = sqrt(gm/cd) tanh(sqrt(gcd/m)t)")
    print("Let g = 9.81, cd = 0.25, m = 68.1")
    print("And initial conditions x = v = 0")

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

    def f1(t, y):
        return y[1]

    def f2(t, y):
        g = 9.81
        cd = 0.25
        m = 68.1
        return g - cd / m * y[1] ** 2
    
    functions = [f1, f2]
    y0 = [0.0, 0.0]
    t, y = function(functions, 0.0, 10.0, y0, h=0.5)
    x = y[0, :]
    v = y[1, :]
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, x, label=lab1)
    plt.plot(t, solX(t), label="Analytical x")
    plt.legend()
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(t, v, label=lab2)
    plt.plot(t, solV(t), label="Analytical v")
    plt.grid()
    figManager = plt.get_current_fig_manager()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    examples_roots()
    examples_optimization()
    examples_quadrature()
    examples_curve_fitting()
    examples_linear_algebra()
    examples_differentiation()
    examples_ODE()
