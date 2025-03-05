import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

main_df = load_panda("HHG", "test/cosine_laser", "debug_data.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e3, band_width=10, field_amplitude=1.6, photon_energy=5.25))

fig, axes = plt.subplots(ncols=2)

for t in range(len(main_df["pick_current_density_z"])):
    axes[0].plot(main_df["k_zs"], main_df["pick_current_density_z"][t])

    x_plu = main_df["pick_current_density_kappa"][t].transpose()[0]
    x_min = main_df["pick_current_density_kappa_minus"][t].transpose()[0]

    y_plu = main_df["pick_current_density_kappa"][t].transpose()[1]      
    y_min = main_df["pick_current_density_kappa_minus"][t].transpose()[1]

    y_plu[y_plu == None] = 0.0
    y_min[y_min == None] = 0.0

    axes[1].plot(x_plu[x_plu > 0], y_plu[x_plu > 0] / x_plu[x_plu > 0] )

axes[0].set_xlabel(legend(r"v_F k_z / \omega_L"))
axes[1].set_xlabel(legend(r"v_F \varkappa / \omega_L"))
axes[0].set_ylabel(legend(r"Occ."))

fig.tight_layout()
plt.show()