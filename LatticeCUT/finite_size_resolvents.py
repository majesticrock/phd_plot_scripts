import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from create_zoom import *
from get_data import *
import continued_fraction_pandas as cf

SYSTEM = "bcc"
G   = 2
E_F = 0
OMEGA_D = 0.02
Ns = [16000, 32000]#2000, 4000, 8000, 

fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0, wspace=0)

w_lin = np.linspace(0, 0.29, 15000, dtype=complex)
w_lin += 1e-4j

for N in Ns:
    main_df = load_panda("lattice_cut", f"./{SYSTEM}", "resolvents.json.gz",
                        **lattice_cut_params(N=N, 
                                             g=G, 
                                             U=0, 
                                             E_F=E_F,
                                             omega_D=OMEGA_D))
    resolvents = cf.ContinuedFraction(main_df, ignore_first=100, ignore_last=130)
    axes[0].plot(w_lin.real, resolvents.spectral_density(w_lin, "amplitude_SC", withTerminator=True), label=f"$N={N}$")
    axes[1].plot(w_lin.real, resolvents.spectral_density(w_lin, "phase_SC",     withTerminator=True), label=f"$N={N}$")
    

for ax in axes:
    resolvents.mark_continuum(ax, label=None) 
    ax.set_ylim(-0.05, 5)
    ax.set_xlim(np.min(w_lin.real), np.max(w_lin.real))
    ax.set_xlabel(r"$\omega$")
    
ax.legend(loc="upper right")
axes[0].set_ylabel(r"$\mathcal{A}_\mathrm{Higgs} (\omega)$")
axes[1].set_ylabel(r"$\mathcal{A}_\mathrm{Phase} (\omega)$")

plt.show()