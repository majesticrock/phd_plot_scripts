import __path_appender as __ap
__ap.append()
from get_data import *
import HeatmapPlotter as hp
from HeatmapPlotter import BUILD_DIR, FILE_ENDING, G_MAX_LOAD, G_MAX_PLOT
from legend import *
import matplotlib.pyplot as plt

N=16000
DOS="fcc"
OMEGA_D=0.05
E_F=0.0

all_data = load_all(f"lattice_cut/{DOS}/N={N}", "resolvents.json.gz")

tasks = [
    (all_data.query(f"E_F == {E_F} & omega_D == {OMEGA_D} & g <= {G_MAX_LOAD}"), "g", legend("g")),
]

fig, axes, plotters, cbar = hp.create_plot(tasks)

plt.show()