import numpy as np
import matplotlib.pyplot as plt
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from create_zoom import *
from get_data import *
from scipy.signal import find_peaks

SYSTEM = 'bcc'
N=16000
params = lattice_cut_params(N=N, 
                            g=1.42,
                            U=0.10, 
                            E_F=-0.5,
                            omega_D=0.02)
main_df = load_panda("lattice_cut", f"./{SYSTEM}", "resolvents.json.gz", **params)

import continued_fraction_pandas as cf
import plot_settings as ps

resolvents = cf.ContinuedFraction(main_df, ignore_first=250, ignore_last=300)
print("Delta_true = ", resolvents.continuum_edges()[0])

fig, ax = plt.subplots()
ax.set_xlabel(r"$\omega [\mathrm{meV}]$")
ax.set_ylabel(r"$\mathcal{A} (\omega) [\mathrm{eV}^{-1}]$")

plotter = ps.CURVEFAMILY(6, axis=ax)
plotter.set_individual_colors("nice")
plotter.set_individual_linestyles(["-", "-.", "--", "-", "--", ":"])

w_lin = np.linspace(-0.005 * main_df["continuum_boundaries"][0], 1.5 * main_df["continuum_boundaries"][0], 1000, dtype=complex)#
w_lin += 1e-4j

A_phase = resolvents.spectral_density(w_lin, "phase_SC",     withTerminator=True)
A_higgs = resolvents.spectral_density(w_lin, "amplitude_SC", withTerminator=True)

plotter.plot(w_lin.real, np.arctan(A_phase), label="Phase")
plotter.plot(w_lin.real, np.arctan(A_higgs), label="Higgs")
ax.set_ylim(-0.5, 5)

denom = resolvents.denominator(w_lin.real, "phase_SC", withTerminator=True).real
plotter.plot(w_lin.real, np.arctan((denom     )))
#plotter.plot(w_lin.real, np.arctan(.1*(denom  )))
#plotter.plot(w_lin.real, np.arctan(.01*(denom )))
#plotter.plot(w_lin.real, 1/denom**2)

resolvents.mark_continuum(ax)



ax.set_xlim(np.min(w_lin.real), np.max(w_lin.real))
ax.legend()
fig.tight_layout()
plt.show()