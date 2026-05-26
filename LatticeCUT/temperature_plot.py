import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

SYSTEM = 'bcc'
DIR = '.'
N=10000

fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True, constrained_layout=True)
axes[-1].set_xlabel("$T / W$")
axes[0].set_ylabel(r"$\Delta / W$")

for j, (ax, U) in enumerate(zip(axes, [0, 0.01, 0.1])):
    for i, G in enumerate([1.5, 2.0]):
        params = lattice_cut_params(N=N, 
                                    g=G,
                                    U=U, 
                                    E_F=-0.5,
                                    omega_D=0.02)
        main_df = load_panda("lattice_cut", f"{DIR}/T_C/{SYSTEM}", "T_C.json.gz", **params)

        ax.plot(main_df["temperatures"], main_df["max_gaps"],  ls="-" , c=f"C{i}", label=f"$g={G}$")
        ax.plot(main_df["temperatures"], main_df["true_gaps"], ls="--", c=f"C{i}")

    ax.set_ylim(0, None)
axes[0].legend()

plt.show()