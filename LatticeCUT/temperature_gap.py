import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

SYSTEM = 'bcc'
DIR = '.'
N=10000
U=0.0
params = lattice_cut_params(N=N, 
                            g=2.0,
                            U=U, 
                            E_F=-0.2,
                            omega_D=0.02)
main_df = load_panda("lattice_cut", f"{DIR}/T_C/{SYSTEM}", "T_C.json.gz", **params)
gap_df = load_panda("lattice_cut", f"{DIR}/T_C/{SYSTEM}", "all_gaps.json.gz", **params)
fig, ax = plt.subplots()

X, Y = np.meshgrid(np.linspace(-1, 1, N, endpoint=True), main_df["temperatures"])
Z = gap_df["finite_gaps"]

min_max = np.max(np.abs(Z))
if U!=0.0:
    cont = ax.pcolormesh(X, Y, Z, norm=TwoSlopeNorm(vcenter=0, vmin=-min_max, vmax=min_max), cmap="seismic")
else:
    cont = ax.pcolormesh(X, Y, np.abs(Z), cmap="viridis")

cbar = fig.colorbar(cont, ax=ax)
cbar.set_label(r"$\Delta$")

ax.set_xlabel(r'$\varepsilon$')
ax.set_ylabel(r'$T$')
fig.tight_layout()
plt.show()