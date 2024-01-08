import numpy as np

def checkEig(Am, lm, vm):
    return np.max(np.abs(Am @ vm - lm * vm))

def EdgingMethod(mat):
    n = mat.shape[0]
    inv = np.zeros((n, n))

    inv[0][0] = 1 / mat[0][0]

    for i in range(1, n):
        u = mat[:i, i].reshape(i, 1)
        v = mat[i, :i].reshape(1, i)
        aii = mat[i, i]

        alph = 1 / (aii - v @ inv[:i, :i] @ u)[0][0]
        r = -alph * (inv[:i, :i] @ u)
        q = -alph * (v @ inv[:i, :i])
        P = inv[:i, :i] - (inv[:i, :i] @ u) @ q

        inv[:i, :i] = P
        inv[:i, i] = r.reshape(i)
        inv[i, :i] = q.reshape(i)
        inv[i, i] = alph


    return inv

def getMax(x):
    m = np.max(np.abs(x[0]))
    l = x[0]

    for i in range(x.shape[0]):
        if np.max(np.abs(x[i])) > m:
            m = np.max(np.abs(x[i]))
            l = x[i]

    return l

def invIterationMethod(A, e):
    n = A.shape[0]
    x = np.ones(n)
    l = 1
    invA = EdgingMethod(A)

    iters = 0
    xlast = x + 1
    while np.max(np.abs(xlast - x)) > e:
        xlast = x
        iters += 1
        l = getMax(x)
        x = invA @ (x / l)

    print(f'iters: {iters} \t\t l: {1 / l} \t\t, x: {x / l}')
    print(np.linalg.eig(A).eigenvectors / np.max(np.abs(np.linalg.eig(A).eigenvectors[:, 1])))
    print(np.linalg.eig(A).eigenvalues)
    print(checkEig(A, 1 / l, x))


e = 1e-13

A = np.array([[2.2, 1, 0.5, 2],
              [1, 1.3, 2, 1],
              [0.5, 2, 0.5, 1.6],
              [2, 1, 1.6, 2]])

invIterationMethod(A, e)
