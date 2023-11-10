import numpy as np

np.set_printoptions(linewidth=200)
np.random.seed(123)

n = 7

mtx = np.array(
    [
        [0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
        [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
        [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
        [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
        [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
        [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
        [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105],
    ]
)

vector = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])


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

    return (D, inv_mtx) if dim == n else inv_mtx


inv_mtx = Inverse(mtx)[0]
inv_full = Inverse(mtx)[1]

ans = (
        np.pad(inv_mtx.dot(vector[: n - 1]), (0, 1))
        + np.append(
    (
            inv_mtx.dot(mtx[: n - 1, n - 1:])
            .dot(mtx[n - 1:, : n - 1])
            .dot(inv_mtx)
            .dot(vector[: n - 1])
            - (inv_mtx.dot(mtx[: n - 1, n - 1:]) * vector[n - 1]).T
    ),
    -mtx[n - 1:, : n - 1].dot(inv_mtx).dot(vector[: n - 1]) + vector[n - 1],
)
        / (
                mtx[n - 1, n - 1]
                - mtx[n - 1:, : n - 1].dot(inv_mtx).dot(mtx[: n - 1, n - 1:])
        )[0]
)

print(abs(mtx.dot(inv_full) - np.identity(n, dtype=float)), "\n")
print(ans, "\n")
print(abs(mtx.dot(ans) - vector), "\n")