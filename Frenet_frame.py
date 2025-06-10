import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Stereographic projection from S^3 to R^3 (project from north pole)
def stereographic_proj(q):
    # q in R^4, q = (q0,q1,q2,q3), q0 != 1 (north pole)
    denom = 1 - q[0]
    return q[1:] / denom

# Parameterize a curve on S^3 (e.g., a loxodrome)
def gamma(s):
    return np.array([np.cos(s), np.sin(s)/np.sqrt(2), np.sin(s)/np.sqrt(2), 0])

# Compute Frenet frame numerically
def numerical_derivative(f, s, h=1e-5):
    return (f(s+h) - f(s-h)) / (2*h)

def normalize(v):
    return v / np.linalg.norm(v)

s_vals = np.linspace(0, 4*np.pi, 500)
curve_3d = np.array([stereographic_proj(gamma(s)) for s in s_vals])

# Select point for Frenet frame visualization
s0 = 2.0
p = stereographic_proj(gamma(s0))

# Tangent vector
t = normalize(numerical_derivative(lambda s: stereographic_proj(gamma(s)), s0))

# Normal vector (approximate)
t1 = normalize(numerical_derivative(lambda s: normalize(numerical_derivative(lambda x: stereographic_proj(gamma(x)), s)), s0))

# Binormal vector
b = np.cross(t, t1)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Plot curve
ax.plot(curve_3d[:,0], curve_3d[:,1], curve_3d[:,2], 'b-', label='Projected path on $S^3$')

# Plot Frenet frame vectors at p
scale = 0.5
ax.quiver(p[0], p[1], p[2], t[0], t[1], t[2], color='r', length=scale, normalize=True, label='$\\mathbf{e}_1$ (Tangent)')
ax.quiver(p[0], p[1], p[2], t1[0], t1[1], t1[2], color='g', length=scale, normalize=True, label='$\\mathbf{e}_2$ (Normal)')
ax.quiver(p[0], p[1], p[2], b[0], b[1], b[2], color='m', length=scale, normalize=True, label='$\\mathbf{e}_3$ (Binormal)')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Stereographic Projection of $S^3$ Path and Frenet Frame')
ax.legend()
plt.show()
