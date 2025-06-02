import matplotlib.pyplot as plt
import numpy as np


Nk = 400
kx = np.linspace(-2, 2, Nk)
ky = np.linspace(-2, 2, Nk)
KX, KY = np.meshgrid(kx, ky)

deltas = [
    np.pi * np.array([1, 0]),
    np.pi * np.array([-0.5, np.sqrt(3)/2]),
    np.pi * np.array([-0.5, -np.sqrt(3)/2])
]

gamma = np.exp(1j  * (deltas[0][0] * KX + deltas[0][1] * KY))
gamma += np.exp(1j * (deltas[1][0] * KX + deltas[1][1] * KY))
gamma += np.exp(1j * (deltas[2][0] * KX + deltas[2][1] * KY))


fig, ax = plt.subplots(figsize=(6, 5))

cont = ax.contourf(KX, KY, np.real(gamma) / np.abs(gamma), cmap='seismic', levels=16)
cbar = fig.colorbar(cont, ax=ax)
cbar.set_label(r"$\frac{\Re [\gamma_k]}{|\gamma_k|}$")

ax.plot(np.array([0., 2., 2., 0, -2., -2., 0.]) / 3., np.array([4., 2., -2., -4., -2., 2., 4.]) / (3 * np.sqrt(3)), c='k')

ax.set_xlabel(r"$k_x / \pi$")
ax.set_ylabel(r"$k_y / \pi$")


fig2, ax2 = plt.subplots(figsize=(6, 5))

cont2 = ax2.contourf(KX, KY, np.imag(gamma) / np.abs(gamma), cmap='seismic', levels=16)
cbar2 = fig2.colorbar(cont2, ax=ax2)
cbar2.set_label(r"$\frac{\Im [\gamma_k]}{|\gamma_k|}$")

ax2.plot(np.array([0., 2., 2., 0, -2., -2., 0.]) / 3., np.array([4., 2., -2., -4., -2., 2., 4.]) / (3 * np.sqrt(3)), c='k')

ax2.set_xlabel(r"$k_x / \pi$")
ax2.set_ylabel(r"$k_y / \pi$")

plt.show()