import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *

N = 200
K_Z = 0.7
SYSTEM = 'bcc'
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
        return (np.cos(0.5 * np.pi * x) * np.cos(0.5 * np.pi * y) * np.cos(0.5 * np.pi * z))
    elif SYSTEM == 'fcc':
        return 0.5 - 0.5 * (np.cos(0.5 * np.pi * x) * np.cos(0.5 * np.pi * y) + np.cos(0.5 * np.pi * z) * np.cos(0.5 * np.pi * y) + np.cos(0.5 * np.pi * x) * np.cos(0.5 * np.pi * z))

from scipy.interpolate import interp1d
gaps = interp1d(main_df['energies'], main_df['Delta'], assume_sorted=True, fill_value='extrapolate', bounds_error=False)

X, Y = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))

fig, ax = plt.subplots()
Z = gaps(dispersion(X, Y, K_Z))
cont = ax.contourf(X, Y, Z, cmap="viridis", levels=41)
cbar = fig.colorbar(cont, ax=ax)
cbar.set_label("$\\Delta (k)$")
ax.set_xlabel("$k_x / \\pi$")
ax.set_ylabel("$k_y / \\pi$")

plt.show()