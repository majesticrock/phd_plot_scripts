import __path_appender as __ap
__ap.append()
from get_data import *
import HeatmapPlotter as hp
from HeatmapPlotter import BUILD_DIR, FILE_ENDING, G_MAX_LOAD, G_MAX_PLOT
from legend import *
import matplotlib.pyplot as plt

N=16000
OMEGA_D=0.02
DOS="bcc"

n_mode = 0

data = load_all(f"lattice_cut/{DOS}/N={N}", "resolvents.json.gz", condition="U=0.1")

tasks = [
    (data.query(f"E_F == {E_F} & omega_D == {OMEGA_D} & g <= {G_MAX_LOAD}"), "g", legend("g"))
        for E_F in [-0.5, -0.4, -0.3]#, -0.2, -0.1
]

fig, axes, plotters, cbar = hp.create_plot(tasks, cf_ignore=[(236, 250), (200, 250), (200, 250), (160, 250), (100, 200)])
for i, E_F in enumerate([-0.5, -0.4, -0.3]):#, -0.2, -0.1
    axes[0][i].set_title(f"$E_\\mathrm{{F}} = {E_F}$")
    
for plotter in plotters:
    plotter.HiggsModes.to_pickle(f"phd_plot_scripts/LatticeCUT/modes/higgs_{n_mode}.pkl")
    plotter.PhaseModes.to_pickle(f"phd_plot_scripts/LatticeCUT/modes/phase_{n_mode}.pkl")
    n_mode +=1

#plt.savefig(f"build/{os.path.basename(__file__).split('.')[0]}.pdf")
plt.show()