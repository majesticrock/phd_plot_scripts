import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from load_full_flow_file import load_full_flow_file
from Momentum import MomentumGrid, Q

data = load_full_flow_file("cpp/NickelCUT/build/test")
ELL_STEP = data["index_of_lowest_ROD"]
L = data["L"]
N = L * L

im_show_kwargs = {
    "origin":         "upper",
    "aspect":         "equal",
    "interpolation" : "nearest",
    "cmap" :          "seismic"
}

q = MomentumGrid(L)
p = MomentumGrid(L)

def DW_channel(V):
    return V[(-p).flat_pos()[:,None], q.flat_pos()[None,:], Q(L).pos]

def BCS_channel(V):
    #return V[p.flat_pos()[:,None], (-p).flat_pos()[:,None], q - p] - V[p.flat_pos()[:,None], (-p).flat_pos()[:,None], q - p].T
    return V[p.flat_pos()[:,None], (-p).flat_pos()[:,None], q - p] + V[q.flat_pos()[:,None], (-q).flat_pos()[:,None], (p - q)].T

channel = BCS_channel

fig_same, ax_same = plt.subplots()

V_same = channel(data["flow_states"][ELL_STEP]["interactions_same_spin"]) * N
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

V_differing = channel(data["flow_states"][ELL_STEP]["interactions_differing_spin"]) * N
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

T = 0.065
tolerance = 1e-8
for l in range(0, ELL_STEP + 1):
    V = channel(
        data["flow_states"][l]["interactions_differing_spin"]
    )
    dispersion = data["flow_states"][l]["dispersion"]

    # g_i = tanh(eps_i / 2T) / (2 eps_i)
    g = np.empty(N)
    for i in range(N):
        eps = dispersion[i]

        if abs(eps) < tolerance:
            g[i] = 1.0 / (4.0 * T)
        else:
            g[i] = np.tanh(eps / (2.0 * T)) / (2.0 * eps)
    sqrt_g = np.sqrt(g)

    # K_tilde_ij = -sqrt(g_i) V_ij sqrt(g_j)
    K_tilde = -sqrt_g[:, None] * V * sqrt_g[None, :]
    # K_tilde should be symmetric/Hermitian
    eigenvalues, eigenvectors = np.linalg.eigh(K_tilde)
    print(
        f"l = {l:4d}, "
        f"lambda_max = {eigenvalues[-1]: .8e}, "
        f"lambda_min = {eigenvalues[0]: .8e}"
    )

plt.show()