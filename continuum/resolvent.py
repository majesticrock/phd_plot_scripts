import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()

from get_data import load_panda, continuum_params
pd_data = load_panda("continuum", "offset_10", "resolvents.json.gz", **continuum_params(12000, 0., 1., 1e-4, 4.25, 5., 10.))

import continued_fraction_pandas as cf
import plot_settings as ps

resolvents = cf.ContinuedFraction(pd_data, ignore_first=5, ignore_last=90)

fig, ax = plt.subplots()
#ax.set_ylim(-0.05, 1)
ax.set_xlabel(r"$\omega [\mathrm{meV}]$")
ax.set_ylabel(r"$\mathcal{A} (\omega) [\mathrm{eV}^{-1}]$")

plotter = ps.CURVEFAMILY(6, axis=ax)
plotter.set_individual_colors("nice")
plotter.set_individual_linestyles(["-", "-.", "--", "-", "--", ":"])

w_lin = np.linspace(-0.01 * pd_data["continuum_boundaries"][0], 1.1 * pd_data["continuum_boundaries"][1], 15000, dtype=complex)
#w_lin = np.linspace(10, 100, 15000, dtype=complex)
w_lin += 1e-4j

plotter.plot(1e3 * w_lin.real, resolvents.spectral_density(w_lin, "phase_SC",     withTerminator=True), label="Phase")
plotter.plot(1e3 * w_lin.real, resolvents.spectral_density(w_lin, "amplitude_SC", withTerminator=True), label="Higgs")

resolvents.mark_continuum(ax, 1e3)

ax.set_xlim(1e3 * np.min(w_lin.real), 1e3 * np.max(w_lin.real))
ax.legend()
fig.tight_layout()
plt.show()