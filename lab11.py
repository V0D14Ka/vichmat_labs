# Ричардсон
import math

import numpy as np
from lab10 import Rotation


def getMaxError(mat, x, vector):
    errors = mat.dot(x) - vector
    return np.max(np.abs(errors))


n = 2
mtx = np.array([[2,1],[1,2]])
vector = np.array([4, 5])

e = 1e-11

n = mtx.shape[0]
ans = np.zeros(n)
eigs = Rotation(n, mtx, 20).solve()[1]

lmin = np.min(eigs)
lmax = np.max(eigs)

tau0 = 2 / (lmin + lmax)
nnn = lmin / lmax

p0 = (1 - nnn) / (1 + nnn)
p1 = (1 - math.sqrt(nnn)) / (1 + math.sqrt(nnn))

maxiters = np.log(2 / e) / np.log(1 / p1)
print(maxiters)

steps = 0
tau = tau0
iters = round(maxiters) + 1

while steps < iters:
    steps += 1
    v = np.cos(2 * steps - 1) * np.pi / (2 * maxiters)
    tau = tau0 / (1 + p0 * v)
    ans -= tau * (mtx.dot(ans) - vector)

    if steps == iters:
        if getMaxError(mtx, ans, vector) > e:
            iters += (round(maxiters)+1)


print("Steps:\n", steps, "\n\n")
print("Answer(x):\n", ans, "\n\n")
print("A*x error:\n", abs(mtx.dot(ans) - vector), "\n\n")
