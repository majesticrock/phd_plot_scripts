import numpy as np
import matplotlib.pyplot as plt
from create_momentum_labels import create_momentum_labels

L=201
T_PRIME=-0.3

k = np.linspace(-np.pi, np.pi, L)
X,Y = np.meshgrid(k, k)
cos_x, cos_y = np.cos(X), np.cos(Y)

energies = -2. * (
    (cos_x + cos_y) 
    + 2 * T_PRIME * cos_x * cos_y
)

fig, ax = plt.subplots(layout="constrained")

cont = ax.contourf(X, Y, energies, cmap="magma", levels=30)
cbar = fig.colorbar(cont, ax=ax)
cbar.set_label(r"$\varepsilon (\mathbf{k})$")
ax.set_ylabel(r"$k_y$")
ax.set_xlabel(r"$k_x$")


twice_k = np.concat([k,k[::-1]])
def constant_energy_line(E):
    arg = (-0.5*E - np.cos(twice_k)) / (1 + 2*T_PRIME*np.cos(twice_k))
    # arccos is valid in this range
    mask = (-1. <= arg) & (arg <= 1.)
    ky = np.empty_like(twice_k)
    ky[~mask] = None
    ky[mask] = np.arccos(arg[mask])
    ky[L:] *= -1.
    return ky

energy_contours = np.array([
    constant_energy_line(EF) for EF in [-2., -1.2, 1.]
])

for ec in energy_contours:
    ax.plot(twice_k, ec)

plt.show()