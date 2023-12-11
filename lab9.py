# Градиентный спуск

import numpy as np

np.set_printoptions(linewidth=200)
np.random.seed(123)

n = 7

mtx = np.array(
    [
        [10, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
        [0.241, 10, 0.139, -0.409, 0.321, 0.0625, 0.101],
        [0.123, -0.239, 10, 0.901, 0.243, 0.819, 0.321],
        [0.413, 0.309, 0.801, 10, 0.423, 0.118, 0.183],
        [0.241, -0.221, -0.243, 0.134, 10, 0.712, 0.423],
        [0.281, 0.525, 0.719, 0.118, -0.974, 10, 0.923],
        [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 10],
    ]
)

vector = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])

tolerance = 1e-5

X = np.zeros(n)
r = vector - mtx.dot(X)
a = np.dot(r, r) / np.dot(mtx.dot(r), r)
steps = 0

while np.linalg.norm(mtx.dot(X) - vector) > tolerance:
    X = X + a * r
    r = r - a * mtx.dot(r)
    a = np.dot(r, r) / np.dot(mtx.dot(r), r)
    steps += 1

ans = X

print("Answer(x):\n", ans, "\n\n")
print("A*x error:\n", abs(mtx.dot(ans) - vector), "\n\n")