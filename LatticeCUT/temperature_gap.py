import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *

SYSTEM = 'single_peak30'
N=8000
main_df = load_panda("lattice_cut", f"./T_C/{SYSTEM}", "T_C.json.gz",
                    **lattice_cut_params(N=N, 
                                         g=1.15,
                                         U=0., 
                                         E_F=-0.5,
                                         omega_D=0.02))

fig, ax = plt.subplots()

X, Y = np.meshgrid(np.linspace(-1, 1, N, endpoint=True), main_df["temperatures"])
Z = np.array([ gap for gap in main_df["finite_gaps"] ])
cont = ax.contourf(X, Y, Z, levels=41)

cbar = fig.colorbar(cont, ax=ax)
cbar.set_label(r"$\Delta$")

ax.set_xlabel(r'$\varepsilon$')
ax.set_ylabel(r'$T$')
fig.tight_layout()
plt.show()