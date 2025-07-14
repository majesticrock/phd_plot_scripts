import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *
import os

figs = []
axes = []
systems = ["bcc", "fcc", "hc", "sc"]

Gs = [0.5, 1, 2]
Ef = 0.5

for system in systems: 
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    figs.append(fig)
    axes.append(ax)
    
    for g in Gs:
        main_df = load_panda("lattice_cut", f"./{system}", "gap.json.gz",
                            **lattice_cut_params(N=2000, 
                                                 g=g, 
                                                 U=0, 
                                                 E_F=Ef,
                                                 omega_D=0.05))
        N = main_df["N"]
        energy_space = main_df["Delta_epsilon"] * (np.linspace(0, N, N)) + main_df["min_energy"]
        ax.plot(energy_space, np.sqrt((energy_space - main_df["E_F"])**2 + main_df["Delta"]**2), label=f"$g={g}$")


    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$\sqrt{(\epsilon-E_\mathrm{F})^2 + \Delta^2}$")
    ax.set_title(system)
    ax.legend(loc="upper right")
    
    fig.savefig(f"phd_plot_scripts/LatticeCUT/build/{os.path.basename(__file__).split('.')[0]}_{system}.svg")

plt.show()