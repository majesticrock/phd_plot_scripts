import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from Momentum import MomentumGrid, Momentum
from load_full_flow_file import load_full_flow_state

data = load_full_flow_state(subdir="", 
                           L=10,
                           T=0,
                           U_0=2,
                           tprime=0.,
                           E_F=0,
                           resume_num="",
                           force_json=False)

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

K0 = MomentumGrid(L)
P0 = MomentumGrid(L)

K = K0[:, None]
P = P0[None, :]

Q = Momentum(L, 0, 0)

V = (
    data["interactions_same_spin"][
        K.pos,
        P.pos,
        Q.pos,
    ]
    - data["interactions_same_spin"][
        P.pos,
        K.pos,
        (K - P - Q).pos,
    ]
    - data["interactions_differing_spin"][
        K.pos,
        P.pos,
        Q.pos,
    ]
) * L*L

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