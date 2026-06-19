import mrock_centralized_scripts.path_appender as __ap
__ap.append()
import mrock_centralized_scripts.LatticeHeatmapPlotter as hp
from get_data import *
from legend import *
import matplotlib.pyplot as plt

N=16000
OMEGA_D=0.02
E_F=-0.5
DOS="bcc"
G=1.5
n_mode = 0

all_data = load_pickle(f"lattice_cut/{DOS}/N={N}", "resolvents.pkl")

tasks = [
    (all_data.query(f"E_F == {E_F} & omega_D == {OMEGA_D} & g == {G} & U>0.5"), "U", legend("U"))
]

import mrock_colormaps as mcm
fig, axes, plotters, cbar = hp.create_plot(tasks, cf_ignore=(250, 260), cmap=mcm.blackidis_white_r, min_exp=-6.5, max_exp=-0.5)

plt.show()