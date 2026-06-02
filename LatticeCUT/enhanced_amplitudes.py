import numpy as np
import matplotlib.pyplot as plt
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

import mrock_centralized_scripts.FullDiagPurger as fdp
from mrock_centralized_scripts.create_figure import create_large_figure
from make_panels_touch import make_panels_touch
from label_axes import label_axes

SYSTEM = 'bcc'
N=16000
OMEGA_D = 0.02
DIR = f"./{SYSTEM}"
E_F=-0.5

fig, axes = create_large_figure(nrows=3, ncols=2, 
                                sharex=True, sharey=True, 
                                layout="constrained", 
                                height_to_width_ratio=0.5)

axes[0,0].set_xlim(-0.3, 0.8)
axes[0,0].set_ylim(-1.1,1.1)
axes[0,0].set_ylabel(r"$\alpha_j^{\mathrm{(enh)}}$")
axes[1,0].set_ylabel(r"$\nu_j^{\mathrm{(enh)}}$")
axes[2,0].set_ylabel(r"$\psi_j^{\mathrm{(enh)}}$")
axes[-1,0].set_xlabel(r"$(\varepsilon - E_\mathrm{F}) / W$")
axes[-1,1].set_xlabel(r"$(\varepsilon - E_\mathrm{F}) / W$")

Gs = [1.8, 2.0]

settings = {
    "U" : [0.0, 0.01],
    "gl.amplitude" : [[None, 0], [1, 0]],
    "gl.phase" : [[0, None], [None, None]],
    "amplitude" : [[0, None], [None, None]],
    "phase" : [[None, 1], [2, 1]]
}

for i in range(len(settings["U"])):
    for j in range(2):
        main_df = load_panda("lattice_cut", DIR, "full_diagonalization.json.gz", **lattice_cut_params(
                                    N=N, 
                                    g=Gs[j],
                                    U=settings["U"][i], 
                                    E_F=E_F,
                                    omega_D=OMEGA_D) )

        purger = fdp.FullDiagPurger(main_df, np.linspace(-1, 1, N) - E_F)
        
        if settings["amplitude"][i][j] is not None:
            purger.plot_amplitude(axes[:2,i], which=settings["amplitude"][i][j], label=f"$g={Gs[j]}$")
            purger.plot_glimmering_phase(axes[2,i], which=settings["gl.phase"][i][j])
        else:
            purger.plot_glimmering_amplitude(axes[:2,i], which=settings["gl.amplitude"][i][j], label=f"$g={Gs[j]}$")
            purger.plot_phase(axes[2,i], which=settings["phase"][i][j])
    axes[0,i].set_title(f"$U={settings['U'][i]}$")

label_axes(axes)
make_panels_touch(fig, axes)
axes[0,0].legend(loc="lower left")

plt.show()