import numpy as np
import matplotlib.pyplot as plt
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

import mrock_centralized_scripts.FullDiagPurger as fdp

SYSTEM = 'bcc'
N=16000
E_F=-0.5
OMEGA_D = 0.02
U=0.0
G = 2
DIR = f"./{SYSTEM}"
params = lattice_cut_params(N=N, 
                            g=G,
                            U=U, 
                            E_F=E_F,
                            omega_D=OMEGA_D)
main_df = load_panda("lattice_cut", DIR, "full_diagonalization.json.gz", **params)

fig, all_axes = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(12, 8))
axes, axes_glimmer = all_axes.T
fig.subplots_adjust(hspace=0)
axes[0].set_ylabel("Higgs")
axes[1].set_ylabel("Occupation")
axes[2].set_ylabel("Phase")
axes[-1].set_xlabel(r"$\varepsilon - \mu$")
axes_glimmer[-1].set_xlabel(r"$\varepsilon - \mu$")

purger = fdp.FullDiagPurger(main_df, np.linspace(-1, 1, N) - main_df["chemical_potential"])
purger.plot_amplitude(axes[:2], label="Result", combined_norm=False)
purger.plot_phase(axes[2]     , label="Result")

print("Amplitude:", purger.amplitude_eigenvalues)
print("Phase:", purger.phase_eigenvalues)

purger.plot_glimmering_amplitude(axes_glimmer[:2], label="Result", combined_norm=False)
purger.plot_glimmering_phase(axes_glimmer[2]     , label="Result")

axes[2].legend(loc="upper right")

plt.show()