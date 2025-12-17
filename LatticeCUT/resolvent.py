import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from create_zoom import *
from get_data import *
from scipy.signal import find_peaks

SYSTEM = 'fcc'
main_df = load_panda("lattice_cut", f"./{SYSTEM}", "resolvents.json.gz",
                    **lattice_cut_params(N=16000, 
                                         g=1.5, 
                                         U=0.0, 
                                         E_F=0,
                                         omega_D=0.02))

import continued_fraction_pandas as cf
import plot_settings as ps

resolvents = cf.ContinuedFraction(main_df, ignore_first=105, ignore_last=500)
print("Delta_true = ", resolvents.continuum_edges()[0])

fig, ax = plt.subplots()
ax.set_xlabel(r"$\omega [\mathrm{meV}]$")
ax.set_ylabel(r"$\mathcal{A} (\omega) [\mathrm{eV}^{-1}]$")

plotter = ps.CURVEFAMILY(6, axis=ax)
plotter.set_individual_colors("nice")
plotter.set_individual_linestyles(["-", "-.", "--", "-", "--", ":"])

w_lin = np.linspace(-0.005 * main_df["continuum_boundaries"][1], 0.29, 5000, dtype=complex)#
w_lin += 1e-5j

A_phase = resolvents.spectral_density(w_lin, "phase_SC",     withTerminator=True)
A_higgs = resolvents.spectral_density(w_lin, "amplitude_SC", withTerminator=True)

plotter.plot(w_lin.real, A_phase, label="Phase")
plotter.plot(w_lin.real, A_higgs, label="Higgs")

resolvents.mark_continuum(ax)

#find_peaks_result = find_peaks(A_phase, prominence=0.05)[0]
#for res in find_peaks_result:
#    x = w_lin[res].real
#    ax.axvline(x, c="red", ls=":")
#find_peaks_result = find_peaks(A_higgs, prominence=0.05)[0]
#for res in find_peaks_result:
#    x = w_lin[res].real
#    ax.axvline(x, c="green", ls=":")

ax.set_ylim(-0.05, 5)
ax.set_xlim(np.min(w_lin.real), np.max(w_lin.real))
ax.legend()
fig.tight_layout()
plt.show()