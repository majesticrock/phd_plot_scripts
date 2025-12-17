import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from create_zoom import *
from get_data import *
import continued_fraction_pandas as cf

SYSTEM = "bcc"
G   = 1.4
E_F = 0.
OMEGA_D = 0.02
N = 16000

fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0, wspace=0)

w_lin = np.linspace(0, 0.2, 15000, dtype=complex)
w_lin += 1e-4j

main_df = load_panda("lattice_cut", f"./{SYSTEM}", "resolvents.json.gz",
                        **lattice_cut_params(N=N, 
                                             g=G, 
                                             U=0, 
                                             E_F=E_F,
                                             omega_D=OMEGA_D))

cmap = plt.get_cmap("seismic")
step = 2
shift_range = np.arange(-10, 10, step)

resolvents = cf.ContinuedFraction(main_df, ignore_first=90, ignore_last=230)
all_higgs = resolvents.spectral_density_varied_depth(w_lin, "amplitude_SC", shift_range)
all_phase = resolvents.spectral_density_varied_depth(w_lin, "phase_SC", shift_range)

avg_spectral_higgs = np.zeros(len(w_lin))
avg_spectral_phase = np.zeros(len(w_lin))
for i, (higgs, phase) in enumerate(zip(all_higgs, all_phase)):
    avg_spectral_higgs += higgs
    avg_spectral_phase += phase
    axes[0].plot(w_lin.real, higgs, c=cmap(i / (len(shift_range) - 1)))
    axes[1].plot(w_lin.real, phase, c=cmap(i / (len(shift_range) - 1)))

avg_spectral_higgs /= len(shift_range)
avg_spectral_phase /= len(shift_range)
axes[0].plot(w_lin.real, avg_spectral_higgs, c="k", ls="--", label="Average")
axes[1].plot(w_lin.real, avg_spectral_phase, c="k", ls="--", label="Average")

deviations = np.zeros(len(all_higgs))
for i, (higgs, phase) in enumerate(zip(all_higgs, all_phase)):
    deviations[i] = np.linalg.norm(higgs - avg_spectral_higgs) + np.linalg.norm(phase - avg_spectral_phase)
best_idx = np.argmin(deviations)
axes[0].plot(w_lin.real, all_higgs[best_idx], c="lime", ls=":", linewidth=3, label="Closest to average")
axes[1].plot(w_lin.real, all_phase[best_idx], c="lime", ls=":", linewidth=3, label="Closest to average")

for ax in axes:
    resolvents.mark_continuum(ax, label=None) 
    ax.set_ylim(-0.05, 5)
    ax.set_xlim(np.min(w_lin.real), np.max(w_lin.real))
    ax.set_xlabel(r"$\omega$")
    
ax.legend(loc="upper right")
axes[0].set_ylabel(r"$\mathcal{A}_\mathrm{Higgs} (\omega)$")
axes[1].set_ylabel(r"$\mathcal{A}_\mathrm{Phase} (\omega)$")

plt.show()