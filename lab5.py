import numpy as np

np.set_printoptions(linewidth=200)
np.random.seed(123)

n = 7

matrix = np.array([
    [2.2, 4, -3, 1.5, 0.6, 2, 0.7],
    [4, 3.2, 1.5, -0.7, -0.8, 3, 1],
    [-3, 1.5, 1.8, 0.9, 3, 2, 2],
    [1.5, -0.7, 0.9, 2.2, 4, 3, 1],
    [0.6, -0.8, 3, 4, 3.2, 0.6, 0.7],
    [2, 3, 2, 3, 0.6, 2.2, 4],
    [0.7, 1, 2, 1, 0.7, 4, 3.2]
]).astype(complex)

vector = np.array([3.2, 4.3, -0.1, 3.5, 5.3, 9.0, 3.7]).astype(complex)
vector_copy = vector.copy()


# Разложение
def Holesky(matrix):
    W = np.zeros((n, n)).astype(complex)
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                W[i][i] = np.sqrt(matrix[i][i] - sum(W[i][k] ** 2 for k in range(i)))
            else:
                W[i][j] = (matrix[i][j] - sum(W[i][k] * W[j][k] for k in range(j))) / W[
                    j
                ][j]

    return W


W = Holesky(matrix)
print(W)

# Прямая подстановка для решения нижнетреугольной системы
for i in range(0, n):
    divisor = W[i][i]
    vector[i] /= divisor
    for j in range(i + 1, n):
        multiplier = W[j][i]
        vector[j] = vector[j] - multiplier * vector[i]

# Обратная подстановка для решения верхнетреугольной системы
for j in range(n - 1, -1, -1):
    divisor = W[j][j]
    vector[j] /= divisor
    for i in range(0, j):
        multiplier = W[j][i]
        vector[i] = vector[i] - multiplier * vector[j]

print(vector, "\n")
print(abs(matrix.dot(vector) - vector_copy))
