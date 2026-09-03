import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from create_momentum_labels import create_momentum_labels
from load_full_flow_file import load_full_flow_file

from Momentum import Momentum

data = load_full_flow_file(subdir="", 
                           L=8,
                           T=0,
                           U_0=-1,
                           tprime=0,
                           E_F=0.01,
                           resume_num="",
                           force_json=False)
ELL_STEP = data["index_of_lowest_ROD"]
L = data["L"]
N = L * L

im_show_kwargs = {
    "origin":         "lower",
    "aspect":         "equal",
    "interpolation" : "nearest",
    "cmap" :          "seismic"
}

p = Momentum(L, 0, L//2)

# density_wave_differing | density_wave_same
# single_particle_energy_differing | single_particle_energy_same
# superconductivity
CHANNEL = "superconductivity"

fig, ax = plt.subplots()
V = data["extracted_channels"][ELL_STEP][CHANNEL][p.pos].reshape(L, L).T * N

vmax = np.max(np.abs(V))
if vmax == 0.0:
    vmax += 0.1
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(V, norm=norm, **im_show_kwargs)

# Set custom tick labels for momentum space
ticks, labels = create_momentum_labels(L)
ax.set_xticks(ticks)
ax.set_xticklabels(labels)
ax.set_yticks(ticks)
ax.set_yticklabels(labels)

ax.set_xlabel(r"$k_y$")
ax.set_ylabel(r"$k_x$")
ax.set_title(CHANNEL)
fig.colorbar(im, label=r"$V(k_x,k_y)$")
fig.tight_layout()

plt.show()