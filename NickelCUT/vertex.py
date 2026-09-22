import numpy as np
import matplotlib.pyplot as plt
from Momentum import Momentum
from load_full_flow_file import load_full_flow_state
from nickel_cut_parameters import FLOW_PARAMETERS

data = load_full_flow_state(subdir="", **FLOW_PARAMETERS, resume_num="", force_json=False)

L = data["L"]
N = L*L
im_show_kwargs = {
    "origin":         "lower",
    "aspect":         "equal",
    "interpolation" : "nearest",
    "cmap" :          "seismic"
}

fig, ax = plt.subplots()

Q = Momentum(L, 0, 0)

V = 2 * (
    data["interactions_differing_spin"][:,:,Q.pos] - 2 * data["interactions_same_spin"][:,:,Q.pos]
) * N

print(np.sum(V) / N)

vmax = np.max(np.abs(V))
if vmax == 0.0:
    vmax += 0.1

im = ax.imshow(V, vmin=-vmax, vmax=vmax, **im_show_kwargs)

ax.set_xlabel(r"$k_i$")
ax.set_ylabel(r"$p_i$")
fig.colorbar(im, label=r"$U(\mathbf{k},\mathbf{p},\mathbf{q})$")
fig.tight_layout()

plt.show()