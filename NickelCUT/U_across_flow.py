import numpy as np
import matplotlib.pyplot as plt
from mrock.get_data import *

data_loader = DataLoader()

data = data_loader.load_panda_file("cpp/NickelCUT/build/test.json.gz")

L = data["L"]
N = L * L

def at_fixed_momentum_transfer(U, Qx, Qy):
    ky_zero_idx = L*L//2
    p = np.arange(L)[:, None]
    q = np.arange(L)[None, :]
    
    V = U[p + ky_zero_idx, q + ky_zero_idx, Qx + L*Qy]
    return V

Q = [0,0]#[L//2, L//2]
ELL_STEP = -2

fig_same, ax_same = plt.subplots()
same_spin = data["flow_states"][ELL_STEP]["interactions_same_spin"]
V_same = at_fixed_momentum_transfer(same_spin, *Q)

im_same = plt.imshow(
    V_same,
    origin="lower",
    aspect="equal",
    interpolation="nearest"
)

ax_same.set_xlabel(r"$k_x'$")
ax_same.set_ylabel(r"$k_x$")
ax_same.set_title("Same spin orientation")
fig_same.colorbar(im_same, label=r"$V(k_x,k_x')$")
fig_same.tight_layout()



fig_differing, ax_differing = plt.subplots()
same_spin = data["flow_states"][ELL_STEP]["interactions_differing_spin"]
V_differing = at_fixed_momentum_transfer(same_spin, *Q)

im_differing = plt.imshow(
    V_differing,
    origin="lower",
    aspect="equal",
    interpolation="nearest"
)

ax_differing.set_xlabel(r"$k_x'$")
ax_differing.set_ylabel(r"$k_x$")
ax_differing.set_title("Differing spin orientation")
fig_differing.colorbar(im_differing, label=r"$V(k_x,k_x')$")
fig_differing.tight_layout()



plt.show()