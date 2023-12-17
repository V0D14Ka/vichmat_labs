# Метод простой итерации

import numpy as np

np.set_printoptions(linewidth=200)
np.random.seed(123)

n = 4

A = np.array(
    [
        [3.82, 1.02, 0.75, 0.81],
        [1.05, 4.53, 0.98, 1.53],
        [0.73, 0.85, 4.71, 0.81],
        [0.88, 0.81, 1.28, 3.50]
    ]
)

b = np.array([15.655, 22.705, 23.480, 16.110])

mtx_copy = A.copy()
vector_copy = b.copy()

e = 1e-15

steps = 0

for i in range(n):
    b[i] /= A[i, i]
    A[i] /= -A[i, i]
    A[i, i] = 0.0


x = np.zeros(n)
x_old = np.ones(n)

while np.linalg.norm(x-x_old) > e:
    x_old = x.copy()
    # print(np.linalg.norm(mtx_copy.dot(x) - vector_copy, np.inf))
    x = b + A.dot(x)
    steps += 1


print("Initial mtx:\n", mtx_copy)
print("Initial vector:\n", vector_copy, "\n\n")
print("Steps:\n", steps)
print("Answer(x):\n", x)
print("A*x error:\n", abs(mtx_copy.dot(x) - vector_copy))
