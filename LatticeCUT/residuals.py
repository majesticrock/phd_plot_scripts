import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from get_data import *

SYSTEM = 'sc'
main_df = load_panda("lattice_cut", f"./test/{SYSTEM}", "residuals.json.gz",
                    **lattice_cut_params(N=4000, 
                                         g=2.5, 
                                         U=0.0, 
                                         E_F=0,
                                         omega_D=0.02), numpy_conversion=False)

fig, axes = plt.subplots(nrows=3, sharex=True)
fig.subplots_adjust(hspace=0)
N = len(main_df["phase.eigenvectors"][0])

epsilon = np.linspace(-1, 1, N)

def add_line(ax, y, **kwargs):
    if len(y) != N:
        return
    ax.plot(epsilon, y, **kwargs)

add_line(axes[0], main_df["amplitude.eigenvectors"][1][:N], label="$j=1$")
add_line(axes[0], main_df["amplitude.eigenvectors"][3][:N], label="$j=2$")
add_line(axes[0], main_df["amplitude.eigenvectors"][5][:N], label="$j=3$")
add_line(axes[0], main_df["amplitude.eigenvectors"][7][:N], label="$j=4$")

add_line(axes[1], main_df["amplitude.eigenvectors"][1][N:], label="$j=1$")
add_line(axes[1], main_df["amplitude.eigenvectors"][3][N:], label="$j=2$")
add_line(axes[1], main_df["amplitude.eigenvectors"][5][N:], label="$j=3$")
add_line(axes[1], main_df["amplitude.eigenvectors"][7][N:], label="$j=4$")

#add_line(axes[2], main_df["phase.eigenvectors"][0])
#add_line(axes[2], main_df["phase.eigenvectors"][1], ls="--")
add_line(axes[2], main_df["phase.eigenvectors"][2])
add_line(axes[2], main_df["phase.eigenvectors"][4])
add_line(axes[2], main_df["phase.eigenvectors"][6])
add_line(axes[2], main_df["phase.eigenvectors"][8])

axes[0].set_ylabel(r"$c^\dagger c^\dagger + c c$")
axes[1].set_ylabel(r"$c^\dagger c$")
axes[2].set_ylabel(r"$c^\dagger c^\dagger - c c$")

axes[-1].set_xlabel(r"$\varepsilon / W$")
axes[0].legend(loc="upper right")

plt.show()