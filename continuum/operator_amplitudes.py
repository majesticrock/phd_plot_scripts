import numpy as np
import matplotlib.pyplot as plt
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

import mrock_centralized_scripts.FullDiagPurger as fdp

DIR = "test"
N=4000

fig, axes = plt.subplots(nrows=3, sharex=True)
fig.subplots_adjust(hspace=0)
axes[0].set_ylabel("Higgs")
axes[1].set_ylabel("Occupation")
axes[2].set_ylabel("Phase")
axes[-1].set_xlabel(r"$\varepsilon - E_\mathrm{F}$")

params = continuum_params(N_k=N, 
                          T=0, 
                          coulomb_scaling=1, 
                          screening=1e-4, 
                          k_F=4.25, 
                          g=2.5, 
                          omega_D=10)
main_df = load_panda("continuum", DIR, "full_diagonalization.json.gz", print_date=False, **params)
gap_df = load_panda("continuum", DIR, "gap.json.gz", print_date=False, **params)
purger = fdp.FullDiagPurger(main_df, np.linspace(-1, 1, N // 2 + 1))

purger.plot_phase(axes[2], label="Result")
purger.plot_amplitude(axes[:2], label="Result")


plt.show()