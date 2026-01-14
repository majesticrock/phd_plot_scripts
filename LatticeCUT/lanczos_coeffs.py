import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()

from get_data import *
fig, ax = plt.subplots()
SYSTEM = 'bcc'
main_df = load_panda("lattice_cut", f"./{SYSTEM}", "resolvents.json.gz",
                    **lattice_cut_params(N=16000, 
                                         g=2.6, 
                                         U=0., 
                                         E_F=0,
                                         omega_D=0.02))

a_inf = (main_df["continuum_boundaries"][0]**2 + main_df["continuum_boundaries"][1]**2) * 0.5
b_inf = (main_df["continuum_boundaries"][1]**2 - main_df["continuum_boundaries"][0]**2) * 0.25
A = main_df["resolvents.phase_SC"][0]["a_i"]
B = main_df["resolvents.phase_SC"][0]["b_i"]

ax.plot(A / a_inf, ls="-", marker='x', label="$a_i$", c="C0")
ax.plot(np.sqrt(B[1:]) / b_inf, ls="-", marker='o', label="$b_i$", c="C1") #The beta_0 is just a normalization constant

ax.axhline(1, ls="--", c="k", label="$\infty$")
ax.legend()
ax.set_xlabel("Iteration $i$")
ax.set_ylabel("Lanczos coefficient")
fig.tight_layout()

plt.show()
