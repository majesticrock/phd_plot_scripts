import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from create_zoom import *
from get_data import *
pd_data = load_panda("lattice_cut", "test/free_electrons3", "resolvents.json.gz",
                    **lattice_cut_params(N=1001, 
                                         g=1, 
                                         U=0, 
                                         band_width=10, 
                                         E_F=5,
                                         omega_D=0.05))

import continued_fraction_pandas as cf
import plot_settings as ps

resolvents = cf.ContinuedFraction(pd_data, ignore_first=5, ignore_last=90)
print("Delta_true = ", resolvents.continuum_edges()[0])

fig, ax = plt.subplots()
ax.set_xlabel(r"$\omega [\mathrm{meV}]$")
ax.set_ylabel(r"$\mathcal{A} (\omega) [\mathrm{eV}^{-1}]$")

plotter = ps.CURVEFAMILY(6, axis=ax)
plotter.set_individual_colors("nice")
plotter.set_individual_linestyles(["-", "-.", "--", "-", "--", ":"])

w_lin = np.linspace(-0.005 * pd_data["continuum_boundaries"][1], 1.1 * pd_data["continuum_boundaries"][1], 15000, dtype=complex)
w_lin += 1e-4j

plotter.plot(w_lin.real, resolvents.spectral_density(w_lin, "phase_SC",     withTerminator=True), label="Phase")
plotter.plot(w_lin.real, resolvents.spectral_density(w_lin, "amplitude_SC", withTerminator=True), label="Higgs")

resolvents.mark_continuum(ax)

ax.set_ylim(-0.05, 0.5)
ax.set_xlim(np.min(w_lin.real), np.max(w_lin.real))
ax.legend()
fig.tight_layout()
plt.show()