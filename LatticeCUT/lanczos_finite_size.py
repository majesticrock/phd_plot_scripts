import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()

from get_data import *
fig, ax = plt.subplots()
SYSTEM = 'fcc'
G=1.2
U=0
E_F=0
OMEGA_D=0.02

for N in [2000, 4000, 8000, 12000, 16000]:
    main_df = load_panda("lattice_cut", f"./{SYSTEM}", "resolvents.json.gz",
                        **lattice_cut_params(N=N, 
                                             g=G, 
                                             U=U, 
                                             E_F=E_F,
                                             omega_D=OMEGA_D))
    a_inf = (main_df["continuum_boundaries"][0]**2 + main_df["continuum_boundaries"][1]**2) * 0.5
    A = main_df["resolvents.amplitude_SC"][0]["a_i"]
    ax.plot(A / a_inf, label=f"$N={N}$")


ax.axhline(1, ls="--", c="k", label="$\infty$")
ax.legend()
ax.set_xlabel("Iteration $i$")
ax.set_ylabel("Lanczos coefficient")
fig.tight_layout()

plt.show()
