import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *

fig, ax = plt.subplots()

SYSTEM = "bcc"#"sc"#"free_electrons3"
main_df = load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz",
                    **lattice_cut_params(N=2000, 
                                         g=5, 
                                         U=0, 
                                         E_F=0,
                                         omega_D=0.05))

energy_space = main_df["Delta_epsilon"] * (np.linspace(0, main_df["N"], main_df["N"]) - main_df["E_F"]) + main_df["min_energy"]
ax.plot(energy_space, main_df["Delta"], "k-")

rho_ax = ax.twinx()
rho_ax.plot(energy_space, main_df["dos"], c="red", ls="--")
rho_ax.tick_params(axis='y', colors='red')
rho_ax.yaxis.label.set_color('red')

ax.set_xlabel(r"$\epsilon - \mu$")
ax.set_ylabel(r"$\Delta$")
rho_ax.set_ylabel(r"$\rho(\epsilon)$")

fig.tight_layout()

plt.show()