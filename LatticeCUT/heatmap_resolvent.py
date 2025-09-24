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

n_mode = 0
all_data = []

for DOS in ["sc", "bcc", "fcc"]:#
    print("Loading:", DOS)
    all_data.append(load_all(f"lattice_cut/{DOS}/N={N}", "resolvents.json.gz"))

tasks = [
    (data.query(f"E_F == {E_F} & omega_D == {OMEGA_D} & g <= {G_MAX_LOAD}"), "g", legend("g"))
        for data in all_data
]

fig, axes, plotters, cbar = hp.create_plot(tasks, cf_ignore=[(220, 250), (236, 250), (220, 250)])
axes[0][0].set_title("sc")
axes[0][1].set_title("bcc")
axes[0][2].set_title("fcc")
    
for plotter in plotters:
    plotter.HiggsModes.to_pickle(f"phd_plot_scripts/LatticeCUT/modes/higgs_{n_mode}.pkl")
    plotter.PhaseModes.to_pickle(f"phd_plot_scripts/LatticeCUT/modes/phase_{n_mode}.pkl")
    n_mode +=1

#plt.savefig(f"build/{os.path.basename(__file__).split('.')[0]}.pdf")
plt.show()