import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()

from get_data import load_panda, hubbard_params
pd_data = load_panda("hubbard/square", "test", "dispersions.json.gz", **hubbard_params(0.0, -2.5, 0.0))

import dispersions_2D as d2d
import plot_settings as ps

resolvents = d2d.Dispersions2D(pd_data)

index = 0

fig, ax = plt.subplots()
ax.set_ylim(-0.05, 1.)
ax.set_xlabel(r"$\omega [t]$")
ax.set_ylabel(r"$\mathcal{A} (\omega) [t^{-1}]$")

plotter = ps.CURVEFAMILY(6, axis=ax)
plotter.set_individual_colors("nice")
plotter.set_individual_linestyles(["-", "-.", "--", "-", "--", ":"])

w_lin = np.linspace(-0.01, pd_data["continuum_boundaries"][1] + 0.3, 5000, dtype=complex)
w_lin += 1e-6j

plotter.plot(w_lin, resolvents.spectral_density(w_lin, "phase_SC_a", index=index), label="Phase")
plotter.plot(w_lin, resolvents.spectral_density(w_lin, "amplitude_SC_a", index=index), label="Higgs")

resolvents.mark_continuum(ax, index=index)

ax.legend()
fig.tight_layout()
plt.show()