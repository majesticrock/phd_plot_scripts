import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation

# Parameters
N = 200
SYSTEM = 'sc'
frames = 100  # number of animation frames

# Load data
main_df = load_panda('lattice_cut', f'./{SYSTEM}', 'gap.json.gz',
                    **lattice_cut_params(N=16000,
                                         g=2.,
                                         U=0,
                                         E_F=-0.5,
                                         omega_D=0.02))

def dispersion(x, y, z):
    if SYSTEM == 'sc':
        return (np.cos(np.pi * x) + np.cos(np.pi * y) + np.cos(np.pi * z)) / 3.
    elif SYSTEM == 'bcc':
        return (np.cos(0.5 * np.pi * x) *
                np.cos(0.5 * np.pi * y) *
                np.cos(0.5 * np.pi * z))
    elif SYSTEM == 'fcc':
        return 0.5 - 0.5 * (
            np.cos(0.5 * np.pi * x) * np.cos(0.5 * np.pi * y) +
            np.cos(0.5 * np.pi * z) * np.cos(0.5 * np.pi * y) +
            np.cos(0.5 * np.pi * x) * np.cos(0.5 * np.pi * z))

# Interpolator for gaps
gaps = interp1d(main_df['energies'], main_df['Delta'], assume_sorted=True,
                fill_value='extrapolate', bounds_error=False)

levels = np.linspace(0, np.max(main_df["Delta"]), 41, endpoint=True)

# Grid
X, Y = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))

# Set up plot
fig, ax = plt.subplots()
initial_Z = gaps(dispersion(X, Y, -1))
cont = ax.contourf(X, Y, initial_Z, cmap="viridis", levels=levels)
cbar = fig.colorbar(cont, ax=ax)
cbar.set_label("$\\Delta (k)$")
ax.set_xlabel("$k_x / \\pi$")
ax.set_ylabel("$k_y / \\pi$")

def update(frame):
    ax.clear()
    k_z = -2 + 4 * frame / frames  # sweep k_z from -1 to 1
    Z = gaps(dispersion(X, Y, k_z))
    cont = ax.contourf(X, Y, Z, cmap="viridis", levels=levels)
    ax.set_xlabel("$k_x / \\pi$")
    ax.set_ylabel("$k_y / \\pi$")
    ax.set_title(f"$k_z/\\pi = {k_z:.2f}$")
    return cont.collections

ani = FuncAnimation(fig, update, frames=frames, blit=False, interval=100)

plt.show()