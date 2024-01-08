import numpy as np

def checkEig(Am, lm, vm):
    return np.max(np.abs(Am @ vm - lm * vm))

def getMax(x):
    m = np.max(np.abs(x[0]))
    l = x[0]

    for i in range(x.shape[0]):
        if np.max(np.abs(x[i])) > m:
            m = np.max(np.abs(x[i]))
            l = x[i]

    return l

def iterationMethod(A, e):
    n = A.shape[0]
    x = np.ones(n)
    l = 1

    iters = 0
    xlast = x + 1
    while np.max(np.abs(xlast - x)) > e:
        xlast = x
        iters += 1
        l = getMax(x)
        x = A @ (x / l)

    print(f'iters: {iters} \t\t l: {l} \t\t, x: {x / l}')
    print(np.linalg.eig(A).eigenvectors[:, 0] / np.max(np.abs(np.linalg.eig(A).eigenvectors[:, 0])))
    print(np.linalg.eig(A).eigenvalues)
    print(checkEig(A, l, x))


e = 1e-13

A = np.array([[-0.1687, 0.353699, 0.008540, 0.733624],
              [0.353699, 0.056519, -0.723182, -0.076440],
              [0.00854, -0.723182, 0.015938, 0.342333],
              [0.733624, -0.07644, 0.342333, -0.045744]])

iterationMethod(A, e)
