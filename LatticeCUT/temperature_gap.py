import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *

SYSTEM = 'bcc'
N=10000
params = lattice_cut_params(N=N, 
                            g=1.85,
                            U=0.2, 
                            E_F=-0.5,
                            omega_D=0.02)
main_df = load_panda("lattice_cut", f"./T_C/{SYSTEM}", "T_C.json.gz", **params)
gap_df = load_panda("lattice_cut", f"./T_C/{SYSTEM}", "all_gaps.json.gz", **params)
fig, ax = plt.subplots()

X, Y = np.meshgrid(np.linspace(-1, 1, N, endpoint=True), main_df["temperatures"])
Z = np.array([ (gap if gap[0] < 0 else -gap) for gap in gap_df["finite_gaps"] ])

print(np.min(Z[0]), np.max(Z[0]))

min_max = np.max(np.abs(Z))
cont = ax.pcolormesh(X, Y, Z, norm=TwoSlopeNorm(vcenter=0, vmin=-min_max, vmax=min_max), cmap="seismic")

cbar = fig.colorbar(cont, ax=ax)
cbar.set_label(r"$\Delta$")

ax.set_xlabel(r'$\varepsilon$')
ax.set_ylabel(r'$T$')
fig.tight_layout()
plt.show()