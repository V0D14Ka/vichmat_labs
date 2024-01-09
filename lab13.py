import numpy as np

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

e = 1e-9

A = np.array([[2.2, 1, 0.5, 2],
              [1, 1.3, 2, 1],
              [0.5, 2, 0.5, 1.6],
              [2, 1, 1.6, 2]])

n = A.shape[0]
x = np.ones(n)
l = 1
invA = EdgingMethod(A)

steps = 0
while np.max(np.abs(A.dot(x) - l * x)) > e:
    steps += 1
    i = np.where(np.abs(x) == np.max(np.abs(x)))[0][0]
    l = x[i]
    x = A.dot((x / l))


print(f'iters: {steps} \n\nl: {1 / l} \n\nx: {x / l}\n\n')
print(
    np.linalg.eig(A).eigenvectors[:, 0]
    / np.max(np.abs(np.linalg.eig(A).eigenvectors[:, 0]))
)
print(np.linalg.eig(invA).eigenvalues)
print(np.max(np.abs(A @ x - l * x)))






