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
import colorspacious as cs
#Ls = [90, 70, 50, 30, 10, 20, 40, 60, 80]
Ls = [80,  60,  40,  20]
dH = -40

colors = [ (1,1,1) ]
for h, L in enumerate(Ls):
    colors.append(cs.cspace_convert((L, 140, 40 + h * dH), "CIELCh", "sRGB1") )
colors.append((0,0,0))
cmap = mcm.perceptual_colormap(colors)

fig, axes, plotters, cbar = hp.create_plot(tasks, cf_ignore=(260, 280), cmap=cmap)
    
#for plotter in plotters:
#    plotter.HiggsModes.to_pickle(f"phd_plot_scripts/LatticeCUT/modes/higgs_{n_mode}.pkl")
#    plotter.PhaseModes.to_pickle(f"phd_plot_scripts/LatticeCUT/modes/phase_{n_mode}.pkl")
#    n_mode +=1

#plt.savefig(f"build/{os.path.basename(__file__).split('.')[0]}.pdf")
plt.show()