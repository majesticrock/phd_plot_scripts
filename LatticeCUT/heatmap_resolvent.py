import __path_appender as __ap
__ap.append()
from get_data import *
import HeatmapPlotter as hp
from HeatmapPlotter import BUILD_DIR, FILE_ENDING, G_MAX_LOAD, G_MAX_PLOT
from legend import *
import matplotlib.pyplot as plt

N=16000
OMEGA_D=0.02
E_F=-0.5
DOS="bcc"
U=0.01
n_mode = 0

all_data = load_all(f"lattice_cut/./{DOS}/N={N}", "resolvents.json.gz", condition=f"U={U}")

tasks = [
    (all_data.query(f"E_F == {E_F} & omega_D == {OMEGA_D} & U == {U}"), "g", legend("g"))
]

fig, axes, plotters, cbar = hp.create_plot(tasks, cf_ignore=(250, 400))
    
#for plotter in plotters:
#    plotter.HiggsModes.to_pickle(f"phd_plot_scripts/LatticeCUT/modes/higgs_{n_mode}.pkl")
#    plotter.PhaseModes.to_pickle(f"phd_plot_scripts/LatticeCUT/modes/phase_{n_mode}.pkl")
#    n_mode +=1

#plt.savefig(f"build/{os.path.basename(__file__).split('.')[0]}.pdf")
plt.show()