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
U=0
n_mode = 0

all_data = load_pickle(f"lattice_cut/{DOS}/N={N}", "resolvents.pkl")

tasks = [
    (all_data.query(f"E_F == {E_F} & omega_D == {OMEGA_D} & U == {U}"), "g", legend("g"))
]

import mrock_colormaps as mcm
l = mcm.perceptual_colormap([
    (0x00 / 255, 0x04 / 255, 0x6a / 255),
    (0x00 / 255, 0x81 / 255, 0xff / 255),
    (1,1,1),
])
r = mcm.perceptual_colormap([
    (1,1,1),
    (0xf0 / 255, 0x47 / 255, 0x00 / 255),
    (0x55 / 255, 0x00 / 255, 0x00 / 255),
])
cmap = mcm.create_diverging_from_existing(l, r)

fig, axes, plotters, cbar = hp.create_plot(tasks, cf_ignore=(236, 250), cmap=cmap)
    
#for plotter in plotters:
#    plotter.HiggsModes.to_pickle(f"phd_plot_scripts/LatticeCUT/modes/higgs_{n_mode}.pkl")
#    plotter.PhaseModes.to_pickle(f"phd_plot_scripts/LatticeCUT/modes/phase_{n_mode}.pkl")
#    n_mode +=1

#plt.savefig(f"build/{os.path.basename(__file__).split('.')[0]}.pdf")
plt.show()