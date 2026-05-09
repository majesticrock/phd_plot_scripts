import numpy as np
import matplotlib.pyplot as plt
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

import mrock_centralized_scripts.FullDiagPurger as fdp

SYSTEM = 'bcc'
E_F=0
OMEGA_D = 0.02
G = 0.9
DIR = f"./{SYSTEM}"
N=16000

fig, axes = plt.subplots(nrows=3, sharex=True)
fig.subplots_adjust(hspace=0)
axes[0].set_ylabel("Higgs")
axes[1].set_ylabel("Occupation")
axes[2].set_ylabel("Phase")
axes[-1].set_xlabel(r"$\varepsilon - \mu$")
axes[0].set_xlim(-0.25, 0.25)


params = lattice_cut_params(N=N, 
                            g=G,
                            U=0, 
                            E_F=E_F,
                            omega_D=OMEGA_D)
main_df = load_panda("lattice_cut", DIR, "full_diagonalization.json.gz", print_date=False, **params)
gap_df = load_panda("lattice_cut", DIR, "gap.json.gz", print_date=False, **params)
purger = fdp.FullDiagPurger(main_df, np.linspace(-1, 1, N) - main_df["chemical_potential"])

purger.plot_amplitude(axes[:2], combined_norm=True)
purger.plot_phase(axes[2], label="Result")

print(purger.amplitude_eigenvalues, purger.phase_eigenvalues)

for ax in axes:
    ax.axhline(0, c="k", ls=":")

plt.show()