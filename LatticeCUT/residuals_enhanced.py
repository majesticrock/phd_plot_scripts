import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from get_data import *

SYSTEM = 'bcc'
N=8000
params = lattice_cut_params(N=N, 
                              g=3., 
                              U=0.0, 
                              E_F=-0.5,
                              omega_D=0.02)
main_df = load_panda("lattice_cut", f"./test/{SYSTEM}", "residuals.json.gz", **params, numpy_conversion=False)
gap_df  = load_panda("lattice_cut", f"./test/{SYSTEM}", "gap.json.gz", **params)
resolvent_df = load_panda("lattice_cut", f"./test/{SYSTEM}", "resolvents.json.gz", **params)
epsilon = np.linspace(-1, 1, N)

omega_minus, _ = resolvent_df["continuum_boundaries"]

fig, axes = plt.subplots(nrows=3, sharex=True)
fig.subplots_adjust(hspace=0)

def add_line(ax, y, **kwargs):
    y = np.asarray(y)
    if len(y) != N:
        return
    if abs(np.min(y)) > abs(np.max(y)):
        y = -y
    ax.plot(epsilon, y / np.max(np.abs(y)), **kwargs)

offset = 0
add_line(axes[0], main_df["amplitude.eigenvectors"][offset + 0][:N], label="$j=1$")
add_line(axes[0], main_df["amplitude.eigenvectors"][offset + 1][:N], label="Enhanced")
add_line(axes[0], main_df["amplitude.eigenvectors"][offset + 3][:N], label="$j=2$")
add_line(axes[0], gap_df["Delta"], ls="--", c="k", label=r"$\Delta$")

add_line(axes[1], main_df["amplitude.eigenvectors"][offset + 0][N:])
add_line(axes[1], main_df["amplitude.eigenvectors"][offset + 1][N:])
add_line(axes[1], main_df["amplitude.eigenvectors"][offset + 3][N:])
add_line(axes[1], gap_df["Delta"], ls="--", c="k", label=r"$\Delta$")

add_line(axes[2], main_df["phase.eigenvectors"][0])
add_line(axes[2], main_df["phase.eigenvectors"][2])
add_line(axes[2], main_df["phase.eigenvectors"][3])
add_line(axes[2], gap_df["Delta"], ls="--", c="k")

axes[0].set_ylabel(r"$c^\dagger c^\dagger + c c$")
axes[1].set_ylabel(r"$c^\dagger c$")
axes[2].set_ylabel(r"$c^\dagger c^\dagger - c c$")

axes[-1].set_xlabel(r"$\varepsilon / W$")
axes[0].legend(loc="upper right")

plt.show()