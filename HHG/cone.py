import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Grid
x = np.linspace(0, np.pi, 100)
y = np.linspace(0, np.pi, 100)
X, Y = np.meshgrid(x, y)

# Surface 1
Z1 = np.sqrt(np.cos(X)**2 + np.cos(Y)**2) / np.sqrt(2)

# Surface 2: shifted cone
v_F = 1.0 / np.sqrt(2)  # can adjust
X_shifted = X - np.pi/2
Y_shifted = Y - np.pi/2
Z2 = Z1 - v_F * np.sqrt(X_shifted**2 + Y_shifted**2) 

# Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z1, cmap='viridis')
ax.plot_surface(X, Y, Z2, cmap='plasma')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
