import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mrock.get_data import *
from create_momentum_labels import create_momentum_labels

data_loader = DataLoader()

data = data_loader.load_panda_file("cpp/NickelCUT/build/test.json.gz")

L = data["L"]
N = L * L

def at_fixed_momentum_transfer(U, Qx, Qy):
    kx_zero_idx = L//2
    ky_zero_idx = L*L//2
    p = np.arange(L)[:, None]
    q = np.arange(L)[None, :]
    
    V = U[kx_zero_idx, kx_zero_idx, q + p]*N#Qx + L*Qy] * N
    return V

Q = [0,0]
ELL_STEP = data["index_of_lowest_ROD"]

im_show_kwargs = {
    "origin":         "lower",
    "aspect":         "equal",
    "interpolation" : "nearest",
    "cmap" :          "seismic"
}



fig_same, ax_same = plt.subplots()
same_spin = data["flow_states"][ELL_STEP]["interactions_same_spin"]
V_same = at_fixed_momentum_transfer(same_spin, *Q)

vmax_same = np.max(np.abs(V_same))
if vmax_same == 0.0:
    vmax_same +=0.1
norm_same = TwoSlopeNorm(vmin=-vmax_same, vcenter=0, vmax=vmax_same)
im_same = ax_same.imshow(V_same, norm=norm_same, **im_show_kwargs)

# Set custom tick labels for momentum space
ticks, labels = create_momentum_labels(L)
ax_same.set_xticks(ticks)
ax_same.set_xticklabels(labels)
ax_same.set_yticks(ticks)
ax_same.set_yticklabels(labels)

ax_same.set_xlabel(r"$k_y$")
ax_same.set_ylabel(r"$k_x$")
ax_same.set_title("Same spin orientation")
fig_same.colorbar(im_same, label=r"$V(k_x,k_y)$")
fig_same.tight_layout()



fig_differing, ax_differing = plt.subplots()
same_spin = data["flow_states"][ELL_STEP]["interactions_differing_spin"]
V_differing = at_fixed_momentum_transfer(same_spin, *Q)

vmax_differing = np.max(np.abs(V_differing))
if vmax_differing == 0.0:
    vmax_differing +=0.1
norm_differing = TwoSlopeNorm(vmin=-vmax_differing, vcenter=0, vmax=vmax_differing)
im_differing = ax_differing.imshow(V_differing, norm=norm_differing, **im_show_kwargs)

# Set custom tick labels for momentum space
ticks, labels = create_momentum_labels(L)
ax_differing.set_xticks(ticks)
ax_differing.set_xticklabels(labels)
ax_differing.set_yticks(ticks)
ax_differing.set_yticklabels(labels)

ax_differing.set_xlabel(r"$k_y$")
ax_differing.set_ylabel(r"$k_x$")
ax_differing.set_title("Differing spin orientation")
fig_differing.colorbar(im_differing, label=r"$V(k_x,k_y)$")
fig_differing.tight_layout()



plt.show()