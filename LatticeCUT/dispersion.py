import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

Gs = [0.5, 1, 1.5, 2]
Ef = -0.5
system = "bcc"
extra=0.2

for g in Gs:
    main_df = load_panda("lattice_cut", f"./{system}", "gap.json.gz",
                    **lattice_cut_params(N=16000, 
                                         g=g, 
                                         U=0, 
                                         E_F=Ef,
                                         omega_D=0.02))
    energies = main_df["energies"]
    deltas = main_df["Delta"]
    ax.plot(energies, np.sqrt(deltas**2 + (energies - Ef)**2) - np.max(deltas), label=f"$g={g}$")

ax.set_ylabel(r"$E(\varepsilon) - \Delta_\mathrm{max}$")
ax.set_xlabel(r"$\varepsilon$")

plt.show()