# Градиентный спуск

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

# b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])

b = np.array([15.655, 22.705, 23.480, 16.110])

tolerance = 1e-14

X = np.zeros(n)
r = b - A.dot(X)
a = np.dot(r, r) / np.dot(A.dot(r), r)
steps = 0

while np.linalg.norm(A.dot(X) - b) > tolerance:
    # print(np.linalg.norm(A.dot(X) - b))
    X = X + a * r
    r = r - a * A.dot(r)
    a = np.dot(r, r) / np.dot(A.dot(r), r)
    steps += 1

ans = X

print("Answer(x):\n", ans)
print("Steps:", steps, "\n\n")
print("A*x error:\n", abs(A.dot(ans) - b), "\n\n")
