import numpy as np


def find_max_row(matrix, col):
    max_val = abs(matrix[col][col])
    max_row = col

    for i in range(col + 1, len(matrix)):
        if abs(matrix[i][col]) > max_val:
            max_val = abs(matrix[i][col])
            max_row = i

    return max_row


def swap_rows(matrix, row1, row2):
    matrix[row1], matrix[row2] = matrix[row2], matrix[row1]


def gauss(matrix, vector, tolerance=1e-13):
    n = len(matrix)

    for col in range(n):
        max_row = find_max_row(matrix, col)
        swap_rows(matrix, col, max_row)
        swap_rows(vector, col, max_row)

        pivot = matrix[col][col]
        if abs(pivot) < tolerance:
            raise Exception("Matrix is singular or nearly singular. Cannot continue.")

        for row in range(col + 1, n):
            factor = matrix[row][col] / pivot
            for j in range(col, n):
                matrix[row][j] -= factor * matrix[col][j]
            vector[row] -= factor * vector[col]

    x = np.zeros(n)
    for col in range(n - 1, -1, -1):
        if abs(matrix[col][col]) < tolerance:
            raise Exception("Matrix is singular or nearly singular. Cannot continue.")
        x[col] = vector[col] / matrix[col][col]
        for row in range(col - 1, -1, -1):
            vector[row] -= matrix[row][col] * x[col]

    return x


A = np.array([[0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
              [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
              [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
              [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
              [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
              [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
              [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105]])
b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])

tolerance = 1e-13
solution = gauss(A.tolist(), b.tolist(), tolerance)
print("Решение:", solution)

# Проверка на |A*x - b| -> 0
residual = np.linalg.norm(np.dot(A, solution) - b)
print("|A * x - b| =", residual)
