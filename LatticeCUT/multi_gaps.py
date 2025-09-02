import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *
import os

figs = []
axes = []
systems = ["bcc", "fcc", "sc"]

Gs = [0.5, 1, 2]
Ef = -0.5

for system in systems: 
    fig, ax = plt.subplots()
    figs.append(fig)
    axes.append(ax)
    
    for g in Gs:
        main_df = load_panda("lattice_cut", f"test/{system}", "gap.json.gz",
                            **lattice_cut_params(N=2000, 
                                                 g=g, 
                                                 U=0, 
                                                 E_F=Ef,
                                                 omega_D=0.02))
        energy_space = main_df["energies"]
        ax.plot(energy_space, main_df["Delta"], label=f"$g={g}$")

    rho_ax = ax.twinx()
    rho_ax.plot(energy_space, main_df["dos"], c="red", ls=":")
    rho_ax.tick_params(axis='y', colors='red')
    rho_ax.yaxis.label.set_color('red')

    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$\Delta$")
    rho_ax.set_ylabel(r"$\rho(\epsilon)$")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.set_title(system)
    ax.legend(loc="upper right")
    
    #fig.savefig(f"phd_plot_scripts/LatticeCUT/build/{os.path.basename(__file__).split('.')[0]}_{system}.pdf")

plt.show()