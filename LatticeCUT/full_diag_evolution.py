import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from get_data import *

import FullDiagPurger as fdp

PICK = 0
SYSTEM = 'sc'
N = 16000
E_F = 0

epsilon = np.linspace(-1, 1, N) - E_F

fig_wv, axes_wv = plt.subplots(nrows=3, sharex=True)
fig_wv.subplots_adjust(hspace=0)
axes_wv[0].set_ylabel(r"$c_{k\uparrow}^\dagger c_{-k\downarrow}^\dagger + \mathrm{H.c}$")
axes_wv[1].set_ylabel(r"$c_{k\uparrow}^\dagger c_{k\uparrow} + c_{k\downarrow}^\dagger c_{k\downarrow}$")
axes_wv[2].set_ylabel(r"$c_{k\uparrow}^\dagger c_{-k\downarrow}^\dagger - \mathrm{H.c}$")
axes_wv[-1].set_xlabel(r"$\varepsilon - E_\mathrm{F}$")
epsilon = np.linspace(-1, 1, N) - E_F
#Gs = [2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.7, 2.65, 2.8, 2.9, 3.0]
Gs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
cmap = plt.get_cmap('plasma', len(Gs))
for i, g in enumerate(Gs):
    params = lattice_cut_params(N=N, 
                                g=g, 
                                U=0.0, 
                                E_F=E_F,
                                omega_D=0.02)
    main_df = load_panda("lattice_cut", f"./{SYSTEM}", "full_diagonalization.json.gz", **params)
    purger = fdp.FullDiagPurger(main_df, epsilon)
    purger.plot_amplitude(axes_wv[:2], PICK, label=f"$g={g}$", c=cmap(i / len(Gs)))
    purger.plot_phase(axes_wv[2], PICK, label=f"$g={g}$", c=cmap(i / len(Gs)))

axes_wv[0].legend(loc="upper right", ncols=2)
axes_wv[0].set_xlim(-0.2, 0.2)

plt.show()