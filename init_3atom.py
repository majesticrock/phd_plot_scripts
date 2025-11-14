import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Parameters
Delta = 100.0
k_vals = np.linspace(-np.pi, np.pi, 500)

# Function to compute band width of lowest eigenvalue
def lowest_band_width(J, Delta):
    eigvals = np.zeros((3, len(k_vals)))
    for i, k in enumerate(k_vals):
        omega = np.exp(1j * k / 3)
        H = np.array([
            [Delta, J * omega, J * np.conj(omega)],
            [J * np.conj(omega), 0, J * omega],
            [J * omega, J * np.conj(omega), Delta]
        ])
        eigvals[:, i] = np.linalg.eigvalsh(H)
    E_min = np.min(eigvals[0, :])
    E_max = np.max(eigvals[0, :])
    return E_max - E_min - 4  # we want width = 4

# Find J such that width of lowest band = 4
res = root_scalar(lowest_band_width, args=(Delta,), bracket=[0.01, 50], method='bisect')
J_target = res.root
print(f"Adjusted J for width 4: {J_target:.6f}")

# Now compute eigenvalues with this J
eigvals = np.zeros((3, len(k_vals)))
for i, k in enumerate(k_vals):
    omega = np.exp(1j * k / 3)
    H = np.array([
        [Delta, J_target * omega, J_target * np.conj(omega)],
        [J_target * np.conj(omega), 0, J_target * omega],
        [J_target * omega, J_target * np.conj(omega), Delta]
    ])
    eigvals[:, i] = np.linalg.eigvalsh(H)

# Shift so lowest band is centered at 0
E_min = np.min(eigvals[0, :])
E_max = np.max(eigvals[0, :])
shift = - (E_min + E_max) / 2
eigvals_shifted = eigvals + shift

# Plot
fig, ax = plt.subplots()
for n in range(3):
    ax.plot(k_vals / np.pi, eigvals_shifted[n, :], label=f'Band {n+1}')
    
ax.plot(k_vals / np.pi, 2 * np.cos(k_vals), ls="--", label=r"$2 \cos (k)$")
ax.set_xlabel(r'$k / \pi$')
ax.set_ylabel(r'$\varepsilon$')
ax.grid()
ax.legend()
plt.show()