import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()

from get_data import *
SYSTEM = 'bcc'
main_df = load_panda("lattice_cut", f"confined/{SYSTEM}", "resolvents.json.gz",
                    **lattice_cut_params(N=16000, 
                                         g=1.8, 
                                         U=0.1, 
                                         E_F=0,
                                         omega_D=0.02))

a_inf = (main_df["continuum_boundaries"][0]**2 + main_df["continuum_boundaries"][1]**2) * 0.5
b_inf = (main_df["continuum_boundaries"][1]**2 - main_df["continuum_boundaries"][0]**2) * 0.25

A = main_df["resolvents.phase_SC"][0]["a_i"]
B = main_df["resolvents.phase_SC"][0]["b_i"]

fig, ax = plt.subplots()
ax.plot(A, ls="-", marker='x', label="$a_i$")
ax.plot(np.sqrt(B), ls="-", marker='o', label="$b_i$")
ax.axhline(a_inf, linestyle="-" , color="k", label="$a_\\infty$")
ax.axhline(b_inf, linestyle="--", color="k", label="$b_\\infty$")
ax.legend()
ax.set_xlabel("Iteration $i$")
ax.set_ylabel("Lanczos coefficient")
#ax.set_ylim(min(a_inf, b_inf) - 0.2 * max(a_inf, b_inf), 1.2 * max(a_inf, b_inf))
fig.tight_layout()

import os
#plt.savefig(f"python/build/{os.path.basename(__file__).split('.')[0]}.pdf")
plt.show()
