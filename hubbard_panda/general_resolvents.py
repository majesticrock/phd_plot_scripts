import numpy as np
import matplotlib.pyplot as plt

from mrock.get_data import *
data_loader = DataLoader()
pd_data = data_loader.load_panda("hubbard/square", "test", "dispersions.json.gz", **hubbard_params(0.0, -2.5, 0.0))

import mrock_centralized_scripts.dispersions_2D as d2d
from mrock_centralized_scripts.plot_settings import *

resolvents = d2d.Dispersions2D(pd_data)

index = 0

fig, ax = plt.subplots()
ax.set_ylim(-0.05, 1.)
ax.set_xlabel(r"$\omega [t]$")
ax.set_ylabel(r"$\mathcal{A} (\omega) [t^{-1}]$")


w_lin = np.linspace(-0.01, pd_data["continuum_boundaries"][1] + 0.3, 5000, dtype=complex)
w_lin += 1e-6j

ax.plot(w_lin, resolvents.spectral_density(w_lin, "phase_SC_a", index=index), label="Phase")
ax.plot(w_lin, resolvents.spectral_density(w_lin, "amplitude_SC_a", index=index), label="Higgs")

resolvents.mark_continuum(ax, index=index)

ax.legend()
fig.tight_layout()
plt.show()