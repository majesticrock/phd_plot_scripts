import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

main_df = load_panda("HHG", "test", "current_density.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e3, band_width=5, field_amplitude=1.6, photon_energy=5.25))

fig, axes = plt.subplots(ncols=2)    
axes[0].plot(main_df["k_zs"], main_df["picky_z_alpha"] - main_df["picky_z_beta"], label=r"$\alpha$")

axes[1].plot(main_df["kappas"], main_df["picky_kappa_alpha"] - main_df["picky_kappa_beta"], label=r"$\alpha$")

axes[0].set_xlabel(legend(r"v_F k_z / \omega_L"))
axes[1].set_xlabel(legend(r"v_F \kappa / \omega_L"))
axes[0].set_ylabel(legend(r"Occ."))

fig.tight_layout()
plt.show()