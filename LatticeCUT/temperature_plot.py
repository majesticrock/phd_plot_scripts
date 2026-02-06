import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

SYSTEM = 'bcc'
DIR = 'test2'
N=2000
U=0.0

fig, axes = plt.subplots(nrows=2, sharex=True)
axes[-1].set_xlabel("$T / W$")
axes[0].set_ylabel(r"$\Delta$")
axes[1].set_ylabel(r"$\mu$")

params = lattice_cut_params(N=N, 
                            g=2.0,
                            U=U, 
                            E_F=-0.5,
                            omega_D=0.02)
main_df = load_panda("lattice_cut", f"{DIR}/T_C/{SYSTEM}", "T_C.json.gz", **params)

axes[0].plot(main_df["temperatures"], main_df["max_gaps"], label=r"$\Delta_\mathrm{max}$")
axes[0].plot(main_df["temperatures"], main_df["true_gaps"], label=r"$\Delta_\mathrm{true}$")
axes[0].plot(main_df["temperatures"], np.abs(main_df["gaps_at_ef"]), label=r"$\Delta_\mathrm{F}$")
axes[1].plot(main_df["temperatures"], main_df["chemical_potentials"], label=r"$\mu$")

axes[0].legend()
fig.tight_layout()

plt.show()