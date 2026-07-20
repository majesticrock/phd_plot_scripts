import mrock_centralized_scripts.LatticeHeatmapPlotter as hp
from mrock.get_data import *
data_loader = DataLoader()
from mrock_centralized_scripts.legend import  *
import matplotlib.pyplot as plt

N=16000
OMEGA_D=0.02
E_F=-0.5
DOS="bcc"
U=0.1

all_data = data_loader.load_pickle(f"lattice_cut", f"{DOS}/N={N}", "resolvents.pkl")

tasks = [
    (all_data.query(f"E_F == {E_F} & omega_D == {OMEGA_D} & U == {U}"), "g", legend("g"))
]

import mrock_centralized_scripts.mrock_colormaps as mcm
fig, axes, plotters, cbar = hp.create_plot(tasks, cf_ignore=(250, 260), 
                                           cmap=mcm.blackidis_white_r, 
                                           energy_range=(0., 1.),
                                           min_exp=-3,
                                           max_exp=2.1)

plt.show()