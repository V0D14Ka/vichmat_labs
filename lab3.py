import numpy as np

n = 7
matrix = np.array([[0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
              [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
              [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
              [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
              [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
              [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
              [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105]])
matrix_copy = matrix.copy()
vector = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])
vector_copy = vector.copy()

divisor = matrix[0, 0]
matrix[0, :] /= divisor
vector[0] /= divisor
k = 0

# Цикл построчно
while k < n - 1:

    # Делаем 0 под главной диагональю
    for j in range(0, k + 1):
        multiplier = matrix[k + 1, j]
        matrix[k + 1, :] -= multiplier * matrix[j, :]
        vector[k + 1] -= multiplier * vector[j]

    # Делаем 1 на главной диагонали
    divisor = matrix[k + 1, k + 1]
    matrix[k + 1, :] /= divisor
    vector[k + 1] /= divisor

    # Получаем 0 над главной диагональю
    for j in range(0, k + 1):
        multiplier = matrix[j, k + 1]
        matrix[j] -= multiplier * matrix[k + 1, :]
        vector[j] -= multiplier * vector[k + 1]
    k += 1

print(matrix)

print("Vector = ", vector, "\n")
print("Погрешность:\n", abs(matrix_copy.dot(vector) - vector_copy))
