import numpy as np

A = np.array([[0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
              [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
              [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
              [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
              [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
              [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
              [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105]])
b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])
n = len(A)


def Gauss(A, b, n):
    iter = 0
    while iter < n:
        maxx = A[iter, iter]
        i = iter
        j = 0

        # find absolute max element of the subarray
        for l in range(iter, n):
            for k in range(n):
                if abs(maxx) < abs(A[l, k]):
                    maxx = A[l, k]
                    i = l
                    j = k

        # swap rows
        A[[iter, i]] = A[[i, iter]]
        b[iter], b[i] = b[i], b[iter]

        # Gaussian elimination
        for l in range(iter + 1, n):
            m = A[l, j] / A[iter, j]
            A[l] -= m * A[iter]
            b[l] -= m * b[iter]

        A[iter] /= maxx
        b[iter] /= maxx

        iter += 1

    # back substitution
    ans = np.zeros(n)
    for k in range(1, n + 1):
        for l in range(n):
            if A[n - k, l] == 1:
                ans[l] = b[n - k]
                i = l

        for k in range(n):
            b[k] -= ans[i] * A[k, i]
            A[k, i] = 0

    return ans


x = Gauss(A, b, n)
print(x)
print((np.abs(np.dot(A, x) - b)))
