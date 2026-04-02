import numpy as np
import matplotlib.pyplot as plt
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

import mrock_centralized_scripts.FullDiagPurger as fdp

SYSTEM = 'sc'
N=4000
E_F=0.0
OMEGA_D = 0.02
G = 0.3
DIR = f"./{SYSTEM}"
params = lattice_cut_params(N=N, 
                            g=G,
                            U=0, 
                            E_F=E_F,
                            omega_D=OMEGA_D)
main_df = load_panda("lattice_cut", DIR, "full_diagonalization.json.gz", **params)
gap_df = load_panda("lattice_cut", DIR, "gap.json.gz", **params)

fig, axes = plt.subplots(nrows=3, sharex=True)
fig.subplots_adjust(hspace=0)
axes[0].set_ylabel("Higgs")
axes[1].set_ylabel("Occupation")
axes[2].set_ylabel("Phase")
axes[-1].set_xlabel(r"$\varepsilon - E_\mathrm{F}$")
axes[0].set_xlim(-0.05, 0.05)

purger = fdp.FullDiagPurger(main_df, np.linspace(-1, 1, N) - E_F)
purger.plot_amplitude(axes[:2], label="Result", combined_norm=True, which=0)
purger.plot_phase(axes[2]     , label="Result", which=0)

eps = np.linspace(-1, 1, N) - E_F
Delta = gap_df["Delta"]
E = np.sqrt(eps**2 + Delta**2)

test_pc = (Delta**2 / E**2)
test_pc /= np.max(test_pc)
axes[0].plot(eps, test_pc / np.sqrt(np.max(np.abs(purger.amplitude_eigenvectors[0]))), c="k", ls="--", label=r"$\Delta_k^2 / E_k^2$")
axes[2].plot(eps, test_pc, c="k", ls="--")

test_num = -Delta**3 / (eps * E**2)
test_num /= np.max(test_num)
axes[1].plot(eps, test_num, c="k", ls="--", label=r"$\Delta_k^3 / (E_k^2 \varepsilon_k)$")

plt.show()