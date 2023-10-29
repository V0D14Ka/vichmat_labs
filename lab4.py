# LU разложение

import numpy as np

np.set_printoptions(linewidth=200)

n = 4

matrix = np.array(
    [
        [4.31, 0.26, 0.61, 0.27],
        [0.26, 2.32, 0.18, 0.34],
        [0.61, 0.18, 3.20, 0.31],
        [0.27, 0.34, 0.31, 5.17],
    ]
)

vector = np.array([1.02, 1.00, 1.34, 1.27])
vector_copy = vector.copy()


def lu(matrix):
    L = np.zeros((n, n))
    np.fill_diagonal(L, 1)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            U[i][j] = matrix[i][j]

            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]

        for j in range(i + 1, n):
            L[j][i] = matrix[j][i]

            for k in range(i):
                L[j][i] -= L[j][k] * U[k][i]
            L[j][i] /= U[i][i]

    return L, U


L, U = lu(matrix)

for i in range(0, n):
    for j in range(i + 1, n):
        multiplier = L[j][i]
        L[j] = L[j, :] - multiplier * L[i, :]
        vector[j] = vector[j] - multiplier * vector[i]

for i in range(n - 1, -1, -1):
    divisor = U[i][i]
    U[i, :] /= divisor
    vector[i] /= divisor
    for j in range(0, i):
        multiplier = U[j][i]
        U[j] = U[j] - multiplier * U[i, :]
        vector[j] = vector[j] - multiplier * vector[i]

print("Answer(x):\n", vector, "\n\n")
print("error:\n", abs(matrix.dot(vector) - vector_copy), "\n\n")