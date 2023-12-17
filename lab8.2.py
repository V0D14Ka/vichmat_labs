# Метод relax-chill

import numpy as np

np.set_printoptions(linewidth=200)
np.random.seed(123)
# n = 7
n = 4

# A = np.array(
#     [
#         [10, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
#         [0.241, 10, 0.139, -0.409, 0.321, 0.0625, 0.101],
#         [0.123, -0.239, 10, 0.901, 0.243, 0.819, 0.321],
#         [0.413, 0.309, 0.801, 10, 0.423, 0.118, 0.183],
#         [0.241, -0.221, -0.243, 0.134, 10, 0.712, 0.423],
#         [0.281, 0.525, 0.719, 0.118, -0.974, 10, 0.923],
#         [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 10],
#     ]
# )

A = np.array(
    [
        [3.82, 1.02, 0.75, 0.81],
        [1.05, 4.53, 0.98, 1.53],
        [0.73, 0.85, 4.71, 0.81],
        [0.88, 0.81, 1.28, 3.50]
    ]
)

# b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])

b = np.array([15.655, 22.705, 23.480, 16.110])

e = 1e-14
w = 1.99

steps = 0
x = np.zeros(n)
x_old = np.ones(n)

while np.linalg.norm(x - x_old) > e:
    # print(np.linalg.norm(x - x_old) > e)
    x_old = x.copy()
    for i in range(n):
        x[i] = (1 - w) * x[i] + w * (
                b[i]
                - sum(A[i, j] * x[j] for j in range(0, i))
                - sum(A[i, j] * x[j] for j in range(i + 1, n))
        ) / A[i, i]
    steps += 1

ans = x

print("Initial mtx:\n", A)
print("Initial vector:\n", b)
print("Steps:\n", steps, "\n\n")
print("Answer(x):\n", ans)
print("A*x error:\n", abs(A.dot(ans) - b))
