import numpy as np

np.set_printoptions(linewidth=200)

e = 1e-8


# A = np.array(
#     [
#         [-0.1687, 0.353699, 0.00854, 0.733624],
#         [0.353699, 0.056519, -0.723182, -0.07644],
#         [0.00854, -0.723182, 0.015938, 0.342333],
#         [0.733624, -0.07644, 0.342333, -0.045744],
#     ]
# )

A = np.array([[2.2, 1, 0.5, 2],
              [1, 1.3, 2, 1],
              [0.5, 2, 0.5, 1.6],
              [2, 1, 1.6, 2]])

n = A.shape[0]
x = np.ones(n)
l = 1

steps = 0
while np.max(np.abs(A.dot(x) - l * x)) > e:
    steps += 1
    i = np.where(np.abs(x) == np.max(np.abs(x)))[0][0]
    l = x[i]
    x = A.dot((x / l))

print(f"steps: {steps} \n\nl: {l} \n\nx: {x/l}\n\n")

print(
    np.linalg.eig(A).eigenvectors[:, 0]
    / np.max(np.abs(np.linalg.eig(A).eigenvectors[:, 0]))
)
print(np.linalg.eig(A).eigenvalues)

print(np.max(np.abs(A @ x - l * x)))