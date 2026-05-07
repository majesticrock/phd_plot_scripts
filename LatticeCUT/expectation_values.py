import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

SYSTEM = 'sc'
N=16000
E_F=0
U=0.0

fig, axes = plt.subplots(ncols=3, figsize=(10, 5),constrained_layout=True,sharex=True)
for G in [3.0, 1.2, 0.3]:
    params = lattice_cut_params(N=N, 
                                g=G,
                                U=U, 
                                E_F=E_F,
                                omega_D=0.02)

    gap_df = load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz", **params)
    epsilon = np.linspace(-1, 1, N, endpoint=True)
    xi = epsilon - gap_df["chemical_potential"]
    Delta = gap_df["Delta"]
    E = np.sqrt(xi**2 + Delta**2)

    y = xi**2 / (2 * E**3)
    axes[0].plot(xi, y / np.max(y), label=G)
    y = 1 / (2 * E)
    axes[1].plot(xi, y / np.max(y))
    y = xi * Delta / (E**3)
    axes[2].plot(xi, y / np.max(y))
    
axes[0].set_xlim(-0.3, 0.3)


axes[0].set_xlabel(r"$\varepsilon - \mu$")
axes[0].set_ylabel(r"$- \dfrac{\partial}{\partial \Re \Delta} \langle \hat{c}_{-k,\downarrow} \hat{c}_{k,\uparrow} \rangle$")
axes[1].set_xlabel(r"$\varepsilon - \mu$")
axes[1].set_ylabel(r"$\mathrm{i} \dfrac{\partial}{\partial \Im \Delta} \langle \hat{c}_{-k,\downarrow} \hat{c}_{k,\uparrow} \rangle$")
axes[2].set_xlabel(r"$\varepsilon - \mu$")
axes[2].set_ylabel(r"$\dfrac{\partial}{\partial \Re \Delta} \langle \hat{n}_{k,\uparrow} + \hat{n}_{-k,\downarrow} \rangle$")

axes[0].legend()


plt.show()