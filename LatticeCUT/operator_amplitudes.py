import numpy as np
import matplotlib.pyplot as plt
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

import mrock_centralized_scripts.FullDiagPurger as fdp

SYSTEM = 'sc'
E_F=0.0
OMEGA_D = 0.02
G = 0.3
DIR = f"./{SYSTEM}"

fig, axes = plt.subplots(nrows=3, sharex=True)
fig.subplots_adjust(hspace=0)
axes[0].set_ylabel("Higgs")
axes[1].set_ylabel("Occupation")
axes[2].set_ylabel("Phase")
axes[-1].set_xlabel(r"$\varepsilon - E_\mathrm{F}$")
axes[0].set_xlim(-0.05, 0.05)

Ns = np.array([4000, 6000, 8000, 16000, 20000, 24000])
C_alphas = np.zeros_like(Ns, dtype=float)

for i, N in enumerate(Ns):
    params = lattice_cut_params(N=N, 
                                g=G,
                                U=0, 
                                E_F=E_F,
                                omega_D=OMEGA_D)
    main_df = load_panda("lattice_cut", DIR, "full_diagonalization.json.gz", print_date=False, **params)
    gap_df = load_panda("lattice_cut", DIR, "gap.json.gz", print_date=False, **params)

    purger = fdp.FullDiagPurger(main_df, np.linspace(-1, 1, N) - E_F)
    
    alpha = purger.amplitude_eigenvectors[0,:N]
    norm = np.max(np.abs(alpha))
    alpha /= norm
    nu = purger.amplitude_eigenvectors[0,N:] / norm
    epsilon = np.linspace(-1, 1, N)
    
    axes[0].plot(epsilon, alpha, label="Result")
    axes[1].plot(epsilon, nu, label="Result")
    purger.plot_phase(axes[2], label="Result", which=0)
    
    C1, C2 = purger.integral_amplitude(0)
    norm = C1 + C2
    C1 /= norm
    C_alphas[i] = C1

print(C_alphas)
fig_i, ax_i = plt.subplots()
from ez_fit import ez_linear_fit
ez_linear_fit(1./Ns, C_alphas, ax_i, np.linspace(0, 1.1/Ns[0], 100))
ax_i.plot(1./Ns, C_alphas, "o", markersize=8)
ax_i.set_xlabel(r"$1/N$")
ax_i.set_ylabel(r"$C_\alpha$")

eps = np.linspace(-1, 1, N) - E_F
Delta = gap_df["Delta"]
E = np.sqrt(eps**2 + Delta**2)

test_pc = (Delta**2 / E**2)
test_pc /= np.max(test_pc)
axes[0].plot(eps, test_pc, c="k", ls="--", label=r"$\Delta_k^2 / E_k^2$")
axes[2].plot(eps, test_pc, c="k", ls="--")

test_num = -Delta**3 / (eps * E**2)
test_num /= np.max(test_num) / np.sqrt(np.max(np.abs(purger.amplitude_eigenvectors[0])))
axes[1].plot(eps, test_num, c="k", ls="--", label=r"$\Delta_k^3 / (E_k^2 \varepsilon_k)$")

plt.show()