import numpy as np
from scipy.linalg import expm

# Example: great circle on S^3
def gamma(s):
    return np.array([np.cos(s), np.sin(s), 0, 0])

def gamma_dot(s):
    return np.array([-np.sin(s), np.cos(s), 0, 0])

def wedge(u, v):
    W = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            W[i,j] = u[i]*v[j] - u[j]*v[i]
    return W

# Magnus expansion up to second order
def magnus_expansion(A_list, ds):
    N = len(A_list)
    # First term: sum of A(t) * ds
    Omega1 = np.zeros_like(A_list[0])
    for A in A_list:
        Omega1 += A * ds
    # Second term: double commutator integral
    Omega2 = np.zeros_like(A_list[0])
    for i in range(N):
        for j in range(i):
            Omega2 += 0.5 * ds**2 * (np.dot(A_list[i], A_list[j]) - np.dot(A_list[j], A_list[i]))
    Omega = Omega1 + Omega2
    return Omega

# Discretize the path
N = 100
s_vals = np.linspace(0, 2*np.pi, N)
ds = s_vals[1] - s_vals[0]
A_list = []

for s in s_vals:
    u = gamma(s)
    du = gamma_dot(s)
    Omega = -0.5 * wedge(u, du)  # Connection matrix at s
    A_list.append(Omega)

Omega = magnus_expansion(A_list, ds)
U = expm(Omega)

print("Magnus expansion holonomy matrix U:")
print(U)
