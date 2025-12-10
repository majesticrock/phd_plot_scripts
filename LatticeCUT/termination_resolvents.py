import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from create_zoom import *
from get_data import *
import continued_fraction_pandas as cf

SYSTEM = "bcc"
G   = 1.5
E_F = 0
OMEGA_D = 0.02
N = 16000

step = 10
ignore_firsts = 250 + np.arange(-50, 100, step)
ignore_lasts  = 250 + np.arange(-50, 100, step) + step

fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0, wspace=0)

w_lin = np.linspace(0, 0.29, 15000, dtype=complex)
w_lin += 1e-4j

main_df = load_panda("lattice_cut", f"./{SYSTEM}", "resolvents.json.gz",
                        **lattice_cut_params(N=N, 
                                             g=G, 
                                             U=0, 
                                             E_F=E_F,
                                             omega_D=OMEGA_D))

cmap = plt.get_cmap("viridis")

avg_spectral_higgs = np.zeros_like(w_lin)
avg_spectral_phase = np.zeros_like(w_lin)

for i, (ifirst, ilast) in enumerate(zip(ignore_firsts, ignore_lasts)):
    resolvents = cf.ContinuedFraction(main_df, ignore_first=ifirst, ignore_last=ilast, messages=False)
    
    spectral_higgs = resolvents.spectral_density(w_lin, "amplitude_SC", withTerminator=True)
    spectral_phase = resolvents.spectral_density(w_lin, "phase_SC",     withTerminator=True)
    
    axes[0].plot(w_lin.real, spectral_higgs, c=cmap(i/(len(ignore_firsts)-1)), label=f"$[{ifirst},{ilast}]$")
    axes[1].plot(w_lin.real, spectral_phase, c=cmap(i/(len(ignore_firsts)-1)), label=f"$[{ifirst},{ilast}]$")
    
    avg_spectral_higgs += spectral_higgs
    avg_spectral_phase += spectral_phase
    
avg_spectral_higgs /= len(ignore_firsts)
avg_spectral_phase /= len(ignore_firsts)

axes[0].plot(w_lin.real, avg_spectral_higgs, c="k", ls="--", label="avg")
axes[1].plot(w_lin.real, avg_spectral_phase, c="k", ls="--", label="avg")

for ax in axes:
    resolvents.mark_continuum(ax, label=None) 
    ax.set_ylim(-0.05, 5)
    ax.set_xlim(np.min(w_lin.real), np.max(w_lin.real))
    ax.set_xlabel(r"$\omega$")
    
#ax.legend(loc="upper right")
axes[0].set_ylabel(r"$\mathcal{A}_\mathrm{Higgs} (\omega)$")
axes[1].set_ylabel(r"$\mathcal{A}_\mathrm{Phase} (\omega)$")

plt.show()