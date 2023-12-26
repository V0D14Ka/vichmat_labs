# Ричардсон

import numpy as np
from lab10 import Rotation


def getMaxError(mat, x, vector):
    errors = mat.dot(x) - vector
    return np.max(np.abs(errors))


n = 2
mtx = np.array([[10.9, 1.2, 2.1, 0.9],
                [1.2, 11.2, 1.5, 2.5],
                [2.1, 1.5, 9.8, 1.3],
                [0.9, 2.5, 1.3, 12.1]])

vector = np.array([-7.0, 5.3, 10.3, 24.6])

e = 1e-3
maxiters = 10

n = mtx.shape[0]
ans = np.zeros(n)
eigs = Rotation(n, mtx, 10).solve()[1]

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

print("Steps:\n", steps, "\n\n")
print("Answer(x):\n", ans, "\n\n")
print("A*x error:\n", abs(mtx.dot(ans) - vector), "\n\n")
