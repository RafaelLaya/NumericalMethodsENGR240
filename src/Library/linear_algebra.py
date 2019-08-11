import numpy as np

"""
        Provides methods for linear algebra applications such as solving linear systems of equations, determinants, 
        PLU, QR, Cholesky decomposition, eigenvalues, and many more
        All row vectors, column vectors and matrices are numpy arrays
"""


def cramers(A, b):
    """ Solves a system of equations using cramer's method: Ax = b
    Args:
        A (n x n floats): A matrix of real values
        b (n x 1 floats): a column vector of real values
    Returns:
      (n x 1 floats): The partial_result of solving Ax = b (which must be a consistent system). the ith value of the single column
      corresponds to the ith variable
    Notes:
        uses functions from scratch unlike cramers_np()
    """
    A = A.astype(float)
    D = determinant(A)
    partial_result = {}
    for row in range(0, A.shape[0]):
        C = A.copy()
        C[:, row] = b.transpose()
        partial_result[row + 1] = determinant(C) / D
    formatted_result = []
    for i in sorted(partial_result.keys()):
        formatted_result.append(partial_result[i])
    return np.array([formatted_result]).transpose()


def cramers_np(A, b):
    """ Solves a system of equations using cramer's method: Ax = b
    Args:
        A (n x n floats): A matrix of real values
        b (n x 1 floats): a column vector of real values
    Returns:
      (n x 1 floats): The partial_result of solving Ax = b (which must be a consistent system). the ith value of the single column
      corresponds to the ith variable
    Notes:
        uses numpy functions to calculate determinants
    """
    detA = np.linalg.det(A)
    if detA == 0:
        return None
    partial_result = {}
    for column in range(0, np.shape(A)[1]):
        C = np.copy(A)
        C[:, [column]] = b
        partial_result[column + 1] = np.linalg.det(C) / detA
    formatted_result = []
    for key in sorted(partial_result.keys()):
        formatted_result.append(partial_result[key])
    return np.array([formatted_result]).transpose()


def matrix_mult(A, B):
    """ Multiplies two matrices of real values
    Args:
        A (m x n floats): Left Matrix
        B (n x p floats): Right Matrix
    Returns:
      (m x p floats): Result A times B
    Raises:
        Exception when dimensions are not valid
    """
    if np.shape(A)[1] != np.shape(B)[0]:
        raise Exception("Invalid dimensions")

    C = np.zeros((np.shape(A)[0], np.shape(B)[1]))

    for i in range(0, np.shape(C)[0]):
        for j in range(0, np.shape(C)[1]):
            C[i][j] = sum_to_n_5(A, B, i, j, np.shape(A)[1])
    return C


def __forward_sum(A, x, i, n):
    """ Helper function that applies a forward sum to Ax = b in order to perform forward substitution
    Args:
        A (m x n floats): a real valued matrix
        X (n x 1 floats): the unknown vector of real values
    Returns:
        (float): the result from forward sum at row i
    """
    result = 0
    for j in range(i):
        result = result + A[i][j] * x[j + 1]
    return - result


def __swap_rows(A, row1, row2):
    """ Swaps row1 and row2 of A
    Args:
        A (n x n floats): A matrix of real values
        row1 (int): the index of one of the pair of rows that will be swapped
        row2 (int): the index of the other row
    Returns:
        A (n x n floats): the result of swapping rows row1 and row2 of A
    """
    for i in range(0, np.shape(A)[1]):
        temp = A[row1][i]
        A[row1][i] = A[row2][i]
        A[row2][i] = temp
    return A


def __swap_cols(A, col1, col2):
    """ Swaps cols col1 and col2 of A
    Args:
        A (n x n floats): A matrix of real values
        col1 (int): the index of one of the pair of cols that will be swapped
        col2 (int): the index of the other col
    Returns:
        A (n x n floats): the result of swapping cols col1 and col2 of A
    """
    for i in range(0, np.shape(A)[1]):
        temp = A[i][col1]
        A[i][col1] = A[i][col2]
        A[i][col2] = temp
    return A


def gauss_elimination(A, b):
    """ Solves Ax = b by use of gauss elimination. A must be square
    Args:
        A (n x n floats): A matrix of real values
        b (n x 1 floats): The result vector in Ax = b
    Returns:
        (n x 1 floats): The estimated solution to Ax = b
    """
    C = np.hstack((A, b))
    return __gauss_elimination_helper(C, np.shape(C)[0], np.shape(C)[1])


def __back_subs(A, b):
    """ performs back substitution to solve Ax = b where A is upper triangular
    Args:
        A (n x n floats): The real valued matrix such that Ax = b
        b (n x 1 floats): The column vector (Ax = b)
    Returns:
        The solution for Ax = b
    Notes:
        This is used when solving a system using PLU factorization
    """
    partial_result = {}
    A = np.hstack((A, b))
    m = np.shape(A)[0]
    n = np.shape(A)[1]
    partial_result[m] = A[m - 1][n - 1] / A[m - 1][n - 2]
    for i in range(m - 2, -1, -1):
        partial_result[i + 1] = (A[i][n - 1] + sum_to_n_4(A, partial_result, i, n)) / A[i][i]
    formatted_result = []
    for key in sorted(partial_result.keys()):
        formatted_result.append(partial_result[key])

    return np.transpose(np.array([formatted_result]))


def __forward_subs(A, b):
    """ performs forward substitution to solve Ax = b where A is lower triangular
    Args:
        A (n x n floats): The real valued matrix such that Ax = b
        b (n x 1 floats): The column vector (Ax = b)
    Returns:
        The solution for Ax = b
    Notes:
        This is used when solving a system using PLU factorization
    """
    partial_result = {}
    A = np.hstack((A, b))
    m = np.shape(A)[0]
    n = np.shape(A)[1]
    partial_result[1] = A[0][n - 1] / A[0][0]
    for i in range(1, m, 1):
        partial_result[i + 1] = (A[i][n - 1] + __forward_sum(A, partial_result, i, n)) / A[i][i]
    formatted_result = []
    for key in sorted(partial_result.keys()):
        formatted_result.append(partial_result[key])

    return np.transpose(np.array([formatted_result]))


def pluSolver(P, L, U, b):
    """ Solves Ax = b when A has been decomposed into A = P @ L @ U
    Args:
        (P: m x m floats, L: m x m floats, U: m x m floats): The matrices such that PA = LU
    Returns:
        The solution for Ax = b
    """
    d = P @ b
    y = __forward_subs(L, d)
    return __back_subs(U, y)


def pluDecomp(A):
    """ Decomposes the matrix A into the product of a permutation matrix, a lower triangular matrix and an upper
    triangular matrix. All matrices are real valued.
    Args:
        A (m x m floats): Matrix to decompose into A = P @ L @ U.
    Returns:
      (P: m x m floats, L: m x m floats, U: m x m floats): The matrices such that PA = LU
    """
    U = A.copy()
    U = A.astype(float)
    m = np.shape(U)[0]
    P = np.eye(m)
    L = np.eye(m)

    for i in range(0, m):
        max_abs_num = 0
        max_abs_index = i
        for j in range(i, m):
            if np.math.fabs(U[j][i]) > np.math.fabs(max_abs_num):
                max_abs_num = U[j][i]
                max_abs_index = j
        if max_abs_index != i:
            U = __swap_rows(U, i, max_abs_index)
            P = __swap_rows(P, i, max_abs_index)
            L = __swap_cols(L, i, max_abs_index)
            L = __swap_rows(L, i, max_abs_index)

        for j in range(i + 1, m):
            L[j][i] = U[j][i] / U[i][i]
            U = __add_row_multiple(U, j, i, - U[j][i] / U[i][i])

    return P, L, U


def __gauss_elimination_helper(A, m, n):
    """ Solves Mx = b by use of gauss elimination. A rrepresents [M b]
    Args:
        A (n x (n+1) floats): The augmented matrix [M b] in Mx = b
        m (int): The number of rows
        n (int); the number of columns (in total)
    Returns:
        (n x 1 floats): The estimated solution to Ax = b
    """
    det = 1.0
    A = A.astype(float)
    partial_result = {}
    for i in range(0, m):
        max_abs_num = 0
        max_abs_index = i
        for j in range(i, m):
            if np.math.fabs(A[j][i]) > np.math.fabs(max_abs_num):
                max_abs_num = A[j][i]
                max_abs_index = j
        A = __swap_rows(A, i, max_abs_index)
        if (i != max_abs_index):
            det *= -1

        for j in range(i + 1, m):
            A = __add_row_multiple(A, j, i, - A[j][i] / A[i][i])

    partial_result[m] = A[m - 1][n - 1] / A[m - 1][n - 2]
    for i in range(m - 2, -1, -1):
        partial_result[i + 1] = (A[i][n - 1] + sum_to_n_4(A, partial_result, i, n)) / A[i][i]
    formatted_result = []
    for key in sorted(partial_result.keys()):
        formatted_result.append(partial_result[key])

    for i in range(m):
        det *= A[i][i]

    return np.transpose(np.array([formatted_result])), det


def determinant_gauss_elim(A, tol=0.00001):
    """ calculates determinant of the real valued square matrix A
    Args:
        A (n x n floats): The real valued matrix whose determinant is desired
        tol (float): tolerance of the estimated error
    Returns:
        (float): the estimated determinant of the matrix A
    Notes: This is less efficient than determinant() since this actually fully solves the system
    """
    det = gauss_elimination(A, np.zeros((np.shape(A)[0], 1)))[1]
    if (np.math.fabs(det) < tol):
        print("Matrix may be singular:", det)
    return det


def determinant(A, tol=0.00001):
    """ calculates determinant of the real valued square matrix A through row operations and using the fact that
    the determinant of a triangular matrix is the product of the values on the main diagonal
       Args:
           A (n x n floats): The real valued matrix whose determinant is desired
           tol (float): tolerance of the estimated error
       Returns:
           (float): the estimated determinant of the matrix A
       """
    m = np.shape(A)[0]
    det = 1.0
    A = A.astype(float)
    for i in range(0, m):
        max_abs_num = 0
        max_abs_index = i
        for j in range(i, m):
            if np.math.fabs(A[j][i]) > np.math.fabs(max_abs_num):
                max_abs_num = A[j][i]
                max_abs_index = j
        A = __swap_rows(A, i, max_abs_index)
        if (i != max_abs_index):
            det *= -1
        if A[i][i] == 0:
            break

        for j in range(i + 1, m):
            A = __add_row_multiple(A, j, i, - A[j][i] / A[i][i])

    for i in range(m):
        det *= A[i][i]

    if np.math.fabs(det) < tol:
        print("Matrix may be singular: ", det)
    return det


def cholesky_dec(A):
    """ Does cholesky decomposition on A so that A = U^t * U where U is upper triangular
    Args:
        A (n x n floats): A matrix of real values that is positive definite
    Returns:
        (n x n floats): The matrix U such that A = transpose(U) * U (where the operator * is matrix multiplication)
    raises:
        Exception when the algorithm fails (likely because A is not positive definite)
    """
    try:
        n = np.shape(A)[0]
        U = np.zeros((n, n))
        for i in range(0, n):
            result = A[i][i]
            for k in range(0, i):
                result = result - U[k][i] ** 2
            if result < 0:
                raise Exception("Not valid matrix")
            U[i][i] = np.sqrt(result)

            for j in range(i + 1, n):
                result = A[i][j]
                for k in range(0, i):
                    result = result - U[k][i] * U[k][j]
                U[i][j] = result / U[i][i]
        return U
    except:
        raise Exception("Not valid matrix")


def inverse_lu(A):
    """ Finds the inverse of A through PLU decomposition
    Args:
        A (n x n floats): An invertible matrix of real values
    Returns:
        (n x n floats): The estimated inverse of A
    """
    P, L, U = pluDecomp(A)
    I = np.eye(np.shape(A)[0])
    m = np.shape(A)[0]
    result = []
    for i in range(m):
        b = I[:, i]
        result.append(pluSolver(P, L, U, np.reshape(b, (m, 1))))
    return np.reshape(np.transpose(np.array(result)), np.shape(A))


def inverse_given_lu(P, L, U):
    """ Finds the inverse of a matrix A through PLU decomposition
    Args:
        P (n x n floats): The permutation matrix P in PA = LU
        L (n x n floats): The lower triangular matrix in PLU decomposition
        U (n x n floats): The upper triangular matrix in PLU decomposition
    Returns:
        (n x n floats): The estimated inverse of A
    """
    I = np.eye(np.shape(P)[0])
    m = np.shape(P)[0]
    result = []
    for i in range(m):
        b = I[:, i]
        result.append(pluSolver(P, L, U, np.reshape(b, (m, 1))))
    return np.reshape(np.transpose(np.array(result)), np.shape(P))


def inverse_elim(A):
    """ Finds the inverse of a matrix A through gaussian elimination
    Args:
        A (n x n floats): The matrix whose inverse is desired
    Returns:
        (n x n floats): The estimated inverse of A
    """
    m = np.shape(A)[0]
    A = A.astype(float)
    result = np.eye(m)
    for i in range(0, m):
        max_abs_num = 0
        max_abs_index = i
        for j in range(i, m):
            if np.math.fabs(A[j][i]) > np.math.fabs(max_abs_num):
                max_abs_num = A[j][i]
                max_abs_index = j
        A = __swap_rows(A, i, max_abs_index)
        result = __swap_rows(result, i, max_abs_index)

        fact = A[i][i]
        for j in range(0, m):
            result[i][j] = result[i][j] / fact
            A[i][j] = A[i][j] / fact

        for j in range(i + 1, m):
            result = __add_row_multiple(result, j, i, - A[j][i] / A[i][i])
            A = __add_row_multiple(A, j, i, - A[j][i] / A[i][i])

        for j in range(0, i):
            result = __add_row_multiple(result, j, i, - A[j][i] / A[i][i])
            A = __add_row_multiple(A, j, i, - A[j][i] / A[i][i])

    return result


def norm_euclidean(X):
    """ Finds the euclidean norm of vector X
    Args:
        X (n x 1 floats): The vector whose euclidean norm is desired
    Returns:
        (float): The euclidean norm of X
    """
    return np.sqrt(np.sum(X ** 2))


def norm_frobenius(A):
    """ Finds the frobenius norm of vector X
    Args:
        X (n x 1 floats): The vector whose euclidean norm is desired
    Returns:
        (float): The frobenius norm of X
    """
    A = A.astype(float)
    n = np.shape(A)[0]
    result = 0.0
    for i in range(n):
        for j in range(n):
            result = result + A[i][j] ** 2
    return np.sqrt(result)


def norm_p(x, p):
    """ Finds the p-order norm of vector X
    Args:
        X (n x 1 floats): The vector whose euclidean norm is desired
    Returns:
        (float): The p-order norm norm of X
    """
    x = x.astype(float)
    result = 0.0
    n = np.shape(x)[0]
    for i in range(n):
        result = result + np.math.fabs(x[i]) ** p
    return np.sqrt(result)


def norm_uniform(x):
    """ Finds the uniform norm of vector X
    Args:
        X (n x 1 floats): The vector whose euclidean norm is desired
    Returns:
        (float): The uniform norm of X
    """
    result = x[0]
    for number in x:
        result = max(np.math.fabs(number), np.math.fabs(result))
    return result


def norm_column_sum(A):
    """ Finds the column sum norm of matrix A
    Args:
        A(n x n floats): The matrix whose norm is wanted
    Returns:
        (float): The column sum norm of A
    """
    result = 0.0
    n = np.shape(A)[0]
    for col in range(n):
        column_sum = 0.0
        for row in range(n):
            column_sum = column_sum + np.math.fabs(A[row][col])
        result = max(result, column_sum)
    return result


def norm_row_sum(A):
    """ Finds the column row norm of matrix A
    Args:
        A(n x n floats): The matrix whose norm is wanted
    Returns:
        (float): The row sum norm of A
    """
    result = 0.0
    n = np.shape(A)[1]
    m = np.shape(A)[0]
    for i in range(m):
        row_sum = 0.0
        for j in range(n):
            row_sum = row_sum + np.math.fabs(A[i][j])
        result = max(row_sum, result)
    return result


def norm_matrix(A, which):
    """ Finds the norm of matrix A given a type of norm
    Args:
        A(n x n floats): The matrix whose norm is wanted
        which (String): Select which norm to use
    Returns:
        (float): The desired norm of A
    Raises:
        Exception when which is an invalid value. "fro", "row", col", "eig" are valid
    """
    if which == "fro":
        return norm_frobenius(A)
    elif which == "row":
        return norm_row_sum(A)
    elif which == "col":
        return norm_column_sum(A)
    elif which == "eig":
        return norm_p2(A)
    raise Exception("Bad matrix norm selection. Available: fro, row, col, eig")


def condition(A, norm):
    """ Calculates the condition number of the matrix A given a norm type
    Args:
        A(n x n floats): The matrix whose norm is wanted
        norm (String): The desired norm. Valid values are "fro", "row", "col", "eig"
    Returns:
        (float): The condition number of A using the given norm
    """
    return norm_matrix(A, norm) * norm_matrix(inverse_elim(A), norm)


def gauss_seidel(A, b, tol=0.001, relax=1.0):
    """ Solves Ax = b through gauss-seidel/jacobi iterative method
    Args:
        A(n x n floats): A real valued matrix that is diagonally dominant
        b(n x 1 floats): The column vector of Ax = b
        tol (float): the tolerance of the estimated error
        relax (float): The coefficient of relaxation for gauss-seidel
    Returns:
        (n x 1 floats): The estimated solution to Ax=b
    Raises:
        Exception when the matrix is not diagonally dominant
    """
    A = A.astype(float)
    b = b.astype(float)
    m = np.shape(A)[0]
    C = np.zeros(np.shape(A))
    d = b / np.diagonal(A).reshape(np.shape(b))

    if (not is_diagonally_dominant(A)):
        raise Exception("Matrix must be diagonally dominant.")

    for i in range(m):
        for j in range(m):
            if i != j:
                C[i][j] = A[i][j] / A[i][i]

    x_new = np.zeros((m, 1))
    while True:
        x_old = x_new.copy()

        for i in range(m):
            x_new[i][0] = float(d[i][0] - np.transpose(C[i, :].reshape(np.shape(b))) @ x_old)

        x_new = relax * x_new + (1 - relax) * x_old
        error = np.abs((x_new - x_old) / x_new) * 100.0
        if (error <= tol).all():
            break
    return x_new


def is_diagonally_dominant(A):
    """ Checks whether the matrix A is diagonally dominant
    Args:
        A(n x n floats): A real valued matrix that will be checked
    Returns:
        (Boolean): True if A is diagonally dominant, otherwise False
    """
    m = np.shape(A)[0]
    for i in range(m):
        sum = 0.0
        for j in range(m):
            if i != j:
                sum = sum + np.math.fabs(A[i][j])
        if (np.math.fabs(A[i][i]) < sum):
            return False
    return True


def norm_p2(A):
    """ Finds the p-norm of order 2 of A
    Args:
        A(n x n floats): The matrix whose norm is wanted
    Returns:
        (float): The p-2 norm of A
    """
    return np.sqrt(max(eigQR(np.transpose(A) @ A)[0]))


def eig_power_method(A, tol=0.001):
    """ Finds a real eigenvalue and an eigenvector of A
    Args:
        A(n x n floats): The matrix whose eigenvector/eigenvalue pair is wanted.
    Returns:
        (eigen_val: float, vector_old: column of floats): this is an estimated eigenvalue, eigenvector pair of A
        (eigen_val: float, None): When an error occurs and an exception is not raised by a zero division
    """
    A = A.astype(float)
    m = np.shape(A)[0]

    error = tol + 1
    vector_old = np.ones((m, 1))
    eigen_val = None
    while (error >= tol):
        while ((np.abs(A @ vector_old) < 0.00001).all()):
            vector_old = np.random.randint(low=0, high=200, size=(m, 1)).astype(float)
        vector_new = A @ vector_old

        if eigen_val != None:
            error = np.math.fabs((max(vector_new, key=abs) - eigen_val) / max(vector_new, key=abs)) * 100.0
        eigen_val = max(vector_new, key=abs)
        vector_old = vector_new / max(vector_new, key=abs)
    return eigen_val, vector_old


def eigenvals_M(A, tol=0.001):
    """ Finds distinct real eigenvalues of the matrix A
    Args:
        A(n x n floats): The matrix whose norm is wanted
        tol (float): the tolerance of the estimated error
    Returns:
        (numpy array of floats): a list with the found eigenvalues of A
    """
    m = np.shape(A)[0]
    L = []
    for i in range(m):
        val, vec = eig_power_method(A, tol)
        L.append(val)
        p = 0
        val = vec[0][0]
        for j in range(max(np.shape(vec))):
            if np.math.fabs(vec[j][0]) > np.math.fabs(val):
                p = j
                val = vec[j][0]
        A = A - (1.0 / (val)) * (vec) @ np.reshape(A[p][:], (1, m))
    return np.array(sorted(L)).reshape(m, 1)


def QR_decomposition(A):
    """ Calculates the QR factorization of A through Householder reflections
    Args:
        A(n x n floats): The matrix whose factorization is desired
    Returns:
        (Q: n x n floats, R: n x n floats): The orthogonal matrix Q and upper triangular matrix R such that A = QR
    """
    A = A * 1.0
    m = np.shape(A)[0]
    Q = np.eye(m)
    for i in range(m):
        a_i = A[i:, i].reshape(m - i, 1)
        e_i = np.eye(m - i)[0:, 0].reshape(m - i, 1)
        u_i = a_i + np.sign(a_i[0][0]) * np.sqrt(np.sum(a_i ** 2)) * e_i
        v_i = u_i / np.sqrt(np.sum(u_i ** 2))
        H_i = np.eye(m - i) - 2 * v_i @ np.transpose(v_i)
        I_i = np.eye(m)
        I_i[i:, i:] = H_i
        H_i = I_i
        Q = Q @ H_i
        A = H_i @ A
    return Q, A


def __add_row_multiple(A, row1, row2, lam):
    """ Adds a multiple lam of the row with index row2 to the row with index row1
    Args:
        A(n x n floats): The matrix that has these rows
        row1 (int): the index of the row (a multiple of the row with index row2 will be added here)
        row2 (int): the index of the row that will be multiplied by lam and added to the row of index row1
    Returns:
        A (n x n floats): the result of adding the multiple lam of the row with index row2 to the row with index row1
    """
    for i in range(0, np.shape(A)[1]):
        A[row1][i] += lam * A[row2][i]
    return A


def tridiagonal_solver(A, b):
    """ Solves a tridiagonal system Ax=b
    Args:
        A(n x n floats): A tridiagonal matrix of real values
        b(n x 1 floats): the column vector of Ax=b
    Returns:
        (n x 1 floats): the result of solving the equation Ax=b
    """
    C = np.hstack((A, b))
    return __tridiag_solver_helper(C, np.shape(C)[0], np.shape(C)[1])


def __tridiag_solver_helper(A, m, n):
    """ Solves a tridiagonal system Ax=b
    Args:
        A(n x n floats): A tridiagonal matrix of real values
        b(n x 1 floats): the column vector of Ax=b
    Returns:
        (n x 1 floats): the result of solving the equation Ax=b
    """
    det = 1.0
    A = A.astype(float)
    partial_result = {}
    for i in range(0, m):

        for j in range(i + 1, min(i + 2, m)):
            A = __add_row_multiple(A, j, i, - A[j][i] / A[i][i])

    partial_result[m] = A[m - 1][n - 1] / A[m - 1][n - 2]
    for i in range(m - 2, -1, -1):
        partial_result[i + 1] = (A[i][n - 1] + sum_to_n_4(A, partial_result, i, min(i + 3, n))) / A[i][i]
    formatted_result = []
    for key in sorted(partial_result.keys()):
        formatted_result.append(partial_result[key])

    return np.transpose(np.array([formatted_result]))


def eigQR(A, tol=0.00001):
    """ Estimates distinct real eigenvalues and eigenvectors of the matrix A through the basic version of the QR algorithm
    Args:
        A (n x n floats): the matrix whose eigenvalues and vectors are desired
        tol (float): tolerance of the estimated error
    Returns:
        (vals: numpy array of floats, vectors: numpy 2d array of floats): vals contains the eigenvalues and vectors contains
        an eigenvector in each column. the i-th eigenvalue corresponds to the i-th eigenvector (i-th column)
    """
    A = A * 1.0
    error = tol + 1
    U = np.eye(np.shape(A)[0])

    while error >= tol:
        oldSubDiag = A.diagonal(offset=-1)
        Q, R = QR_decomposition(A)
        A = R @ Q
        U = U @ Q
        error = np.math.fabs((A.diagonal(offset=-1).sum() - oldSubDiag.sum())) * 100.0

    return np.diagonal(A), U


def sum_to_n_5(A, B, i, j, n):
    result = 0
    for lam in range(0, n):
        result += A[i][lam] * B[lam][j]
    return result


def sum_to_n_4(A, X, i, n):
    result = 0
    for j in range(i + 1, n - 1):
        result = result + A[i][j] * X[j + 1]
    return - result
