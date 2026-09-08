import mrock_centralized_scripts.LatticeHeatmapPlotter as hp
from mrock.get_data import *
data_loader = DataLoader()
from mrock_centralized_scripts.legend import  *
import matplotlib.pyplot as plt

N=16000
OMEGA_D=0.02
E_F=0.
DOS="sc"
U=0.

all_data = data_loader.load_pickle(f"lattice_cut", f"LW/{DOS}/N={N}", "resolvents.pkl")

tasks = [
    (all_data.query(f"E_F == {E_F} & omega_D == {OMEGA_D} & U == {U}"), "g", legend("g"))
]

import mrock_centralized_scripts.mrock_colormaps as mcm
fig, axes, plotters, cbar = hp.create_plot(tasks, cf_ignore=(90, 160), 
                                           cmap=mcm.blackidis_white_r, 
                                           energy_range=(0., 0.33),
                                           min_exp=-3,
                                           max_exp=2.1)

plt.show()