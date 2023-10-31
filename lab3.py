import numpy as np

n = 7
matrix = np.array(
    [
        [2.2, 4, -3, 1.5, 0.6, 2, 0.7],
        [4, 3.2, 1.5, -0.7, -0.8, 3, 1],
        [-3, 1.5, 1.8, 0.9, 3, 2, 2],
        [1.5, -0.7, 0.9, 2.2, 4, 3, 1],
        [0.6, -0.8, 3, 4, 3.2, 0.6, 0.7],
        [2, 3, 2, 3, 0.6, 2.2, 4],
        [0.7, 1, 2, 1, 0.7, 4, 3.2],
    ]
)
matrix_copy = matrix.copy()
vector = np.array([3.2, 4.3, -0.1, 3.5, 5.3, 9, 3.7])
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

    # Делаем 0 на главной диагонали
    divisor = matrix[k + 1, k + 1]
    matrix[k + 1, :] /= divisor
    vector[k + 1] /= divisor

    # Получаем 0 над главной диагональю
    for j in range(0, k + 1):
        multiplier = matrix[j, k + 1]
        matrix[j] -= multiplier * matrix[k + 1, :]
        vector[j] -= multiplier * vector[k + 1]
    k += 1


print("Vector = ", vector, "\n")
print("Погрешность:\n", abs(matrix_copy.dot(vector) - vector_copy))
