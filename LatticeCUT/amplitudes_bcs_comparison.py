import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from get_data import *

import FullDiagPurger as fdp

SYSTEM = 'sc'
N=16000
E_F=0.0
OMEGA_D = 0.02
G = 1
DIR = f"./{SYSTEM}"
params = lattice_cut_params(N=N, 
                            g=G,
                            U=0, 
                            E_F=E_F,
                            omega_D=OMEGA_D)
main_df = load_panda("lattice_cut", DIR, "full_diagonalization.json.gz", **params)
gap_df = load_panda("lattice_cut", DIR, "gap.json.gz", **params)

fig, all_axes = plt.subplots(nrows=3, ncols=2, sharex=True)
axes, axes_bcs = all_axes.T
fig.subplots_adjust(hspace=0)
axes[0].set_ylabel("Higgs")
axes[1].set_ylabel("Occupation")
axes[2].set_ylabel("Phase")
axes[-1].set_xlabel(r"$\varepsilon - E_\mathrm{F}$")
axes[0].set_xlim(-0.05, 0.05)

purger = fdp.FullDiagPurger(main_df, np.linspace(-1, 1, N) - E_F)
purger.plot_amplitude(axes[:2], label="Result")
purger.plot_phase(axes[2]     , label="Result")

eps = np.linspace(-1, 1, N) - E_F
Delta = gap_df["Delta"]
E = np.sqrt(eps**2 + Delta**2)

test_pc = (Delta**2 / E**2)
test_pc /= np.max(test_pc)
axes[0].plot(eps, test_pc, c="k", ls="--", label=r"$\Delta_k^2 / E_k^2$")
axes[2].plot(eps, test_pc, c="k", ls="--")

test_num = -Delta**2 / (eps * E)
test_num /= np.max(test_num)
axes[1].plot(eps, test_num, c="k", ls="--", label=r"$\Delta_k^2 / (E_k \varepsilon_k)$")
axes[0].set_title("CUT")

###########################################################################
DIR = f"test_bcs/{SYSTEM}"
N = 4000
params["N"] = N
main_df = load_panda("lattice_cut", DIR, "full_diagonalization.json.gz", **params)
gap_df = load_panda("lattice_cut", DIR, "gap.json.gz", **params)


purger = fdp.FullDiagPurger(main_df, np.linspace(-1, 1, N) - E_F)
purger.plot_amplitude(axes_bcs[:2], label="Result")
purger.plot_phase(axes_bcs[2]     , label="Result")

eps = np.linspace(-1, 1, N) - E_F
Delta = gap_df["Delta"]
E = np.sqrt(eps**2 + Delta**2)

test_pc = (Delta**2 / E**2)
test_pc /= np.max(test_pc)
axes_bcs[0].plot(eps, test_pc, c="k", ls="--", label=r"$\Delta_k^2 / E_k^2$")
axes_bcs[2].plot(eps, test_pc, c="k", ls="--")

test_num = -Delta**2 / (eps * E)
test_num /= np.max(test_num)
axes_bcs[1].plot(eps, test_num, c="k", ls="-.", label=r"$\Delta_k^2 / (E_k \varepsilon_k)$")

axes_bcs[0].set_title("BCS")
axes_bcs[0].legend(loc="upper right")
axes_bcs[1].legend(loc="upper right")
plt.show()