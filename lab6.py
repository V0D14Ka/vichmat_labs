import numpy as np

np.set_printoptions(linewidth=200)
np.random.seed(123)

n = 7

mtx = np.array([
    [2.2, 4, -3, 1.5, 0.6, 2, 0.7],
    [4, 3.2, 1.5, -0.7, -0.8, 3, 1],
    [-3, 1.5, 1.8, 0.9, 3, 2, 2],
    [1.5, -0.7, 0.9, 2.2, 4, 3, 1],
    [0.6, -0.8, 3, 4, 3.2, 0.6, 0.7],
    [2, 3, 2, 3, 0.6, 2.2, 4],
    [0.7, 1, 2, 1, 0.7, 4, 3.2]
])

vector = np.array([3.2, 4.3, -0.1, 3.5, 5.3, 9.0, 3.7])


def Inverse(matrix):
    dim = matrix.shape[0]

    if dim == 1:
        return 1 / matrix

    A_n1 = matrix[: dim - 1, : dim - 1]
    u_n = matrix[: dim - 1, dim - 1:]
    v_n = matrix[dim - 1:, : dim - 1]
    a_nn = matrix[dim - 1, dim - 1]

    D = Inverse(A_n1)

    alpha = a_nn - v_n.dot(D).dot(u_n)

    e1 = D + (D.dot(u_n).dot(v_n).dot(D)) / alpha
    e2 = -D.dot(u_n) / alpha
    e3 = -v_n.dot(D) / alpha
    e4 = 1 / alpha

    inv_mtx = np.vstack((np.hstack((e1, e2)), np.hstack((e3, e4))))

    return inv_mtx


inv_full = Inverse(mtx)

ans = inv_full.dot(vector)

print(ans, "\n")
print(abs(mtx.dot(ans) - vector), "\n")
