import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *

fig, ax = plt.subplots()

SYSTEM = "fcc"#"simple_cubic"#"free_electrons3"

main_df = load_panda("lattice_cut", f"test/{SYSTEM}", "gap.json.gz",
                    **lattice_cut_params(N=1001, 
                                         g=5, 
                                         U=0, 
                                         band_width=4, 
                                         E_F=0,
                                         omega_D=1))

energy_space = main_df["Delta_epsilon"] * (np.linspace(0, main_df["N"], main_df["N"]) - main_df["E_F"]) + main_df["min_energy"]
ax.plot(energy_space, main_df["Delta"], "k-")

rho_ax = ax.twinx()
rho_ax.plot(energy_space, main_df["dos"], c="pink", ls="--")


ax.set_xlabel(r"$\epsilon - \mu$")
ax.set_ylabel(r"$\Delta$")
rho_ax.set_ylabel(r"$\rho(\epsilon)$")

fig.tight_layout()

plt.show()