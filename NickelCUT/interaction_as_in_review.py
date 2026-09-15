import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from itertools import chain
from load_full_flow_file import load_full_flow_state
from nickel_cut_parameters import FLOW_PARAMETERS

data = load_full_flow_state(subdir="", **FLOW_PARAMETERS, resume_num="", force_json=False)

L = data["L"]

A = np.array([
    x + L*((L//2 - x) % L)
    for x in range(L)
])
B = np.array([
    x + L*((x - L//2) % L)
    for x in range(L)
])
B = np.setdiff1d(B, A, assume_unique=True)

FS_points = np.concatenate((A, B))

fix_Q = (L*L) // 2

im_show_kwargs = {
    "origin":         "lower",
    "aspect":         "equal",
    "interpolation" : "nearest",
    "cmap" :          "seismic"
}

fig, ax = plt.subplots()

perm = np.array([
    ((-x) % L) + L*((-y) % L)
    for y in range(L)
    for x in range(L)
])

V = data["interactions_differing_spin"][:, :, fix_Q] * L*L
vmax = np.max(np.abs(V))
if vmax == 0.0:
    vmax += 0.1
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax.imshow(V, norm=norm, **im_show_kwargs)

ax.set_xlabel(r"$k_i$")
ax.set_ylabel(r"$p_i$")
fig.colorbar(im, label=r"$U(\mathbf{k},\mathbf{p},\mathbf{q})$")
fig.tight_layout()

plt.show()