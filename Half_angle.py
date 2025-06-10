import numpy as np
import matplotlib.pyplot as plt

# Parametrize spherical path (closed loop)
theta = np.linspace(0, 2*np.pi, 200)
phi = np.pi/4 + 0.2*np.sin(3*theta)

# Convert to Cartesian coordinates on S^2
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# Plot sphere surface for context
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
X = np.outer(np.cos(u), np.sin(v))
Y = np.outer(np.sin(u), np.sin(v))
Z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(X, Y, Z, color='cyan', alpha=0.1, edgecolor='none')

# Plot path
ax.plot(x, y, z, 'r-', linewidth=2, label='Path on $S^2$')

# Shade solid angle region (approximate)
verts = np.column_stack((x, y, z))
ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], color='red', alpha=0.3)

ax.set_title('Berry Phase as Half the Solid Angle on $S^2$')
ax.legend()
plt.show()
