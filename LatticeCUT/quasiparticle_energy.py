import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *
import os

figs = []
axes = []
systems = ["sc", "bcc", "fcc"]#, "fcc", "hc", "sc"

Gs = [1.35, 1.4, 1.45]#[1.0, 1.2, 1.3, 1.4, 1.5, 1.6]
Ef = 0.

for system in systems: 
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    figs.append(fig)
    axes.append(ax)
    
    for g in Gs:
        main_df = load_panda("lattice_cut", f"./{system}", "gap.json.gz",
                            **lattice_cut_params(N=16000, 
                                                 g=g, 
                                                 U=0, 
                                                 E_F=Ef,
                                                 omega_D=0.02))
        energy_space = main_df["energies"]
        print(energy_space)
        ax.plot(energy_space, np.sqrt((energy_space - main_df["E_F"])**2 + main_df["Delta"]**2) - np.max(main_df["Delta"]), label=f"$g={g}$")


    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$E(\varepsilon) - \Delta_\mathrm{max}$")
    ax.set_title(system)
    ax.legend(loc="upper right")
    
    #fig.savefig(f"phd_plot_scripts/LatticeCUT/build/{os.path.basename(__file__).split('.')[0]}_{system}.svg")

plt.show()