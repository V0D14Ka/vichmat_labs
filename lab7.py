import numpy as np

n = 7

mtx = np.array(
    [
        [0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
        [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
        [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
        [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
        [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
        [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
        [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105],
    ]
)

vector = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])


def QR(mtx):
    R = np.copy(mtx)
    Q = np.eye(n)

    for k in range(n - 1):
        x = R[k:, k]
        v = np.zeros_like(x)
        v[0] = np.sign(x[0]) * np.linalg.norm(x) + x[0]
        v[1:] = x[1:]

        v = v / np.linalg.norm(v)

        R[k:, k:] = R[k:, k:] - 2 * np.outer(v, np.dot(v, R[k:, k:]))

        Qk = np.eye(n)
        Qk[k:, k:] = Qk[k:, k:] - 2 * np.outer(v, v)
        Q = np.dot(Q, Qk)

    return Q, R


Q, R = QR(mtx)

ans = Q.T.dot(vector)

for i in range(n - 1, -1, -1):
    ans[i] /= R[i, i]
    for j in range(0, i):
        ans[j] = ans[j] - R[j][i] * ans[i]

print("Answer(x):\n", ans, "\n\n")
print("A*x:\n", mtx.dot(ans), "\n\n")
print("A*x error:\n", abs(mtx.dot(ans) - vector), "\n\n")