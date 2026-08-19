import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from load_full_flow_file import load_full_flow_file

data = load_full_flow_file("cpp/NickelCUT/build/test10x10")
ELL_STEP = data["index_of_lowest_ROD"]
L = data["L"]
N = L * L

im_show_kwargs = {
    "origin":         "lower",
    "aspect":         "equal",
    "interpolation" : "nearest",
    "cmap" :          "seismic"
}

fig_same, ax_same = plt.subplots()

V_same = data["flow_states"][ELL_STEP]["interactions_same_spin"][:,:,0] * N
vmax_same = np.max(np.abs(V_same))
if vmax_same == 0.0:
    vmax_same += 0.1
norm_same = TwoSlopeNorm(vmin=-vmax_same, vcenter=0, vmax=vmax_same)

im_same = ax_same.imshow(V_same, norm=norm_same, **im_show_kwargs)

ax_same.set_xlabel(r"$p_i$")
ax_same.set_ylabel(r"$q_i$")
ax_same.set_title("Same spin orientation")
fig_same.colorbar(im_same, label=r"$V(\mathbf{p},\mathbf{q})$")
fig_same.tight_layout()



fig_differing, ax_differing = plt.subplots()

V_differing = data["flow_states"][ELL_STEP]["interactions_differing_spin"][:,:,0] * N
vmax_differing = np.max(np.abs(V_differing))
if vmax_differing == 0.0:
    vmax_differing += 0.1
norm_differing = TwoSlopeNorm(vmin=-vmax_differing, vcenter=0, vmax=vmax_differing)

im_differing = ax_differing.imshow(V_differing, norm=norm_differing, **im_show_kwargs)

ax_differing.set_xlabel(r"$p_i$")
ax_differing.set_ylabel(r"$q_i$")
ax_differing.set_title("Differing spin orientation")
fig_differing.colorbar(im_differing, label=r"$V(\mathbf{p},\mathbf{q})$")
fig_differing.tight_layout()

import numpy.linalg as npalg
for l in range(1, ELL_STEP + 1):
    K = data["flow_states"][l]["interactions_differing_spin"][:,:,0]
    for j in range(N):
        K[:,j] /= 2. * data["flow_states"][l]["dispersion"][j]
        
    eigenvalues, eigenvectors = npalg.eig(K)
    print(eigenvalues.max(), eigenvalues.min())

plt.show()