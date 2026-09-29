import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from create_momentum_labels import create_momentum_labels
from load_full_flow_file import load_full_flow_file
from nickel_cut_parameters import FLOW_PARAMETERS

from Momentum import Momentum

data = load_full_flow_file(**FLOW_PARAMETERS, resume_num="", force_json=False)

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
matrices = [
    flow_data[CHANNEL][p.pos].reshape(L, L) * N #- 2 * flow_data["single_particle_energy_same"][p.pos].reshape(L, L) * N
    for flow_data in data["extracted_channels"]
]

vmax = max(np.max(np.abs(V)) for V in matrices)
if vmax == 0.0:
    vmax += 0.1
im = ax.imshow(matrices[0], vmin=-vmax, vmax=vmax, **im_show_kwargs)

# Set custom tick labels for momentum space
ticks, labels = create_momentum_labels(L)
ax.set_xticks(ticks)
ax.set_xticklabels(labels)
ax.set_yticks(ticks)
ax.set_yticklabels(labels)

ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
title = ax.set_title(CHANNEL)
fig.colorbar(im, label=r"$V(k_x,k_y)$")
fig.tight_layout()

def update(ell_step):
    im.set_data(matrices[ell_step])
    title.set_text(f"{CHANNEL}, ELL_STEP={ell_step}, l={data['l_times'][ell_step]:.3g}")
    return im, title

animation = FuncAnimation(
    fig,
    update,
    frames=len(matrices),
    interval=400,
    blit=False,
    repeat=True,
)

plt.show()