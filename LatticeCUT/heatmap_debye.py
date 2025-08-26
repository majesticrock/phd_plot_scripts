import __path_appender as __ap
__ap.append()
from get_data import *
import HeatmapPlotter as hp
from legend import *
import matplotlib.pyplot as plt

N=16000
E_F=0.0

n_mode = 0

for DOS in ["sc", "bcc", "fcc"]:#
    all_data = load_all(f"lattice_cut/{DOS}/N={N}", "resolvents.json.gz")
    tasks = [
        (all_data.query(f"E_F == {E_F} & g == 1.5 & omega_D < 0.06"), "omega_D", legend(r"\omega_\mathrm{D}")),
    ]

    fig, axes, plotters, cbar = hp.create_plot(tasks, cf_ignore=(100 if DOS=="fcc" else 70, 250), scale_energy_by_gaps=True)
    fig.suptitle(f"{DOS} lattice")
    
    #for plotter in plotters:
    #    plotter.HiggsModes.to_pickle(f"phd_plot_scripts/LatticeCUT/modes/higgs_{n_mode}.pkl")
    #    plotter.PhaseModes.to_pickle(f"phd_plot_scripts/LatticeCUT/modes/phase_{n_mode}.pkl")
    #n_mode +=1

plt.show()