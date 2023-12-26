# Ричардсон

import numpy as np
from lab10 import Rotation


def getMaxError(mat, x, vector):
    errors = mat.dot(x) - vector
    return np.max(np.abs(errors))


mtx = np.array([[2, 1], [1, 2]])
n = 2
vector = np.array([4, 5])
e = 1e-3
maxiters = 10


n = mtx.shape[0]
ans = np.zeros(n)
eigs = Rotation(n, mtx, 10).solve().answer
lmin = np.min(eigs)
lmax = np.max(eigs)


tau0 = 2 / (lmin + lmax)
nnn = lmin / lmax
p0 = (1 - nnn) / (1 + nnn)

steps = 0
tau = tau0
while getMaxError(mtx, ans, vector) > e:
    steps += 1
    v = np.cos(2 * steps - 1) * np.pi / (2 * maxiters)
    tau = tau0 / (1 + p0 * v)

    ans -= tau * (mtx.dot(ans) - vector)

    if steps == maxiters:
        break

print("Initial mtx:\n", mtx, "\n\n")
print("Initial vector:\n", vector, "\n\n")
print("Steps:\n", steps, "\n\n")
print("Answer(x):\n", ans, "\n\n")
print("A*x:\n", mtx.dot(ans), "\n\n")
print("A*x error:\n", abs(mtx.dot(ans) - vector), "\n\n")