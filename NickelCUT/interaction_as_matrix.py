import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from load_full_flow_file import load_full_flow_file

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
    "origin":         "upper",
    "aspect":         "equal",
    "interpolation" : "nearest",
    "cmap" :          "seismic"
}

# density_wave_differing | density_wave_same
# single_particle_energy_differing | single_particle_energy_same
# superconductivity
CHANNEL = "superconductivity"

fig, ax = plt.subplots()

V = data["extracted_channels"][ELL_STEP][CHANNEL] * N
vmax = np.max(np.abs(V))
if vmax == 0.0:
    vmax += 0.1
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax.imshow(V, norm=norm, **im_show_kwargs)

ax.set_xlabel(r"$p_i$")
ax.set_ylabel(r"$q_i$")
ax.set_title(CHANNEL)
fig.colorbar(im, label=r"$V(\mathbf{p},\mathbf{q})$")
fig.tight_layout()

plt.show()