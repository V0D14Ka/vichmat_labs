# Вращения с преградами

import numpy as np

class Rotation:
    def __init__(self, n: int, mtx: np.ndarray, p: int):
        self.k = 0
        self.n = n
        self.mtx = mtx
        self.p = p

    def sign(self, num):
        if num > 0:
            return 1
        return -1

    def index_of_largest(self, matrix):
        largest = abs(matrix[0][1])
        i_l, j_l = 0, 1
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                if abs(matrix[i, j]) > largest:
                    largest = abs(matrix[i, j])
                    i_l, j_l = i, j
        return i_l, j_l

    def check_tol(self, matrix):
        self.tol = np.sqrt(max(abs(np.diag(matrix)))) * (10 ** (-self.k))
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                if abs(matrix[i, j]) >= self.tol:
                    return False
        if self.k < self.p:
            self.k += 1
            return False
        return True

    def solve(self):
        mtx = self.mtx
        steps = 0
        while not self.check_tol(mtx):
            q, p = self.index_of_largest(mtx)

            d = abs(mtx[p, p] - mtx[q, q]) / np.sqrt(
                (mtx[p, p] - mtx[q, q]) ** 2 + 4 * mtx[p, q] ** 2
            )

            c = np.sqrt(0.5 * (1 + d))
            s = self.sign(mtx[p, q] * (mtx[p, p] - mtx[q, q])) * np.sqrt(0.5 * (1 - d))

            R = np.eye(self.n)
            R[p, p] = R[q, q] = c
            R[p, q] = -s
            R[q, p] = s
            mtx = R.T.dot(mtx).dot(R)
            mtx[p, q] = mtx[q, p] = 0
            # print(p, " ", q, "\n", mtx, "\n")
            steps += 1

        return steps, np.diag(mtx)


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    np.random.seed(123)

    n = 4
    p = 16
    mtx = np.array(
        [
            [-0.1687, 0.353699, 0.00854, 0.733624],
            [0.353699, 0.056519, -0.723182, -0.07644],
            [0.00854, -0.723182, 0.015938, 0.342333],
            [0.733624, -0.07644, 0.342333, -0.045744],
        ]
    )
    ans1 = Rotation(n, mtx, p).solve()

    print("Initial mtx:\n", mtx, "\n\n")
    print("Steps:\n", ans1[0], "\n\n")
    print("Eigenvalues:\n", sorted(ans1[1], reverse=True), "\n\n")
