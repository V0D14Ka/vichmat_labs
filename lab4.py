# LU разложение

import numpy as np

n = 7

matrix = np.array([[0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
              [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
              [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
              [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
              [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
              [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
              [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105]])

vector = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])
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

print(L)
print(U)

# Обратная подстановка
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

print("Vector:\n", vector, "\n\n")
print("Погрешность:\n", abs(matrix.dot(vector) - vector_copy), "\n\n")
