import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()

from get_data import load_panda, continuum_params
pd_data = load_panda("continuum", "offset_10", "resolvents.json.gz",
                    **continuum_params(N_k=20000, T=0, coulomb_scaling=0, screening=1e-4, k_F=4.25, g=0.4, omega_D=24))

a_inf = (pd_data["continuum_boundaries"][0]**2 + pd_data["continuum_boundaries"][1]**2) * 0.5
b_inf = (pd_data["continuum_boundaries"][1]**2 - pd_data["continuum_boundaries"][0]**2) * 0.25

A = pd_data["resolvents.phase_SC"][0]["a_i"]
B = pd_data["resolvents.phase_SC"][0]["b_i"]

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
