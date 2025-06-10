import numpy as np
from scipy.linalg import expm

# Define wedge product for vectors in R^4 as 4x4 antisymmetric matrix
def wedge(u, v):
    # u, v are 4D vectors
    W = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            W[i,j] = u[i]*v[j] - u[j]*v[i]
    return W

# Parameterize path gamma(s) on S^3 (great circle)
def gamma(s):
    return np.array([np.cos(s), np.sin(s), 0, 0])

# Derivative of gamma
def gamma_dot(s):
    return np.array([-np.sin(s), np.cos(s), 0, 0])

# Discretize path
N = 100
s_vals = np.linspace(0, 2*np.pi, N)
ds = s_vals[1] - s_vals[0]

# Initialize holonomy as identity
R = np.eye(4)

# Compute path-ordered exponential as product of exponentials
for i in range(N-1):
    u = gamma(s_vals[i])
    du = gamma_dot(s_vals[i])
    Omega = wedge(u, du)  # 4x4 antisymmetric matrix
    # Exponentiate -1/2 * Omega * ds
    R_step = expm(-0.5 * Omega * ds)
    R = R_step @ R

print("Holonomy matrix R after one loop:")
print(R)
