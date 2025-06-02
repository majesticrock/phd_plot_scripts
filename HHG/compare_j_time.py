import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

import current_density_time as cdt

fig, ax = cdt.create_frame()

for i, tau in enumerate([10, 50, 100, 500, 1000, -1]):
    #main_df = load_panda("HHG", "cl1_2_cycle_shift/cosine_laser/PiFlux", "current_density.json.gz", 
    #                 **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=tau))
    main_df = load_panda("HHG", "2_cycle/cosine_laser/Honeycomb", "current_density.json.gz", 
                     **hhg_params(T=300, E_F=118, v_F=1e6, band_width=3300, field_amplitude=1.6, photon_energy=5.25, decay_time=tau))
    label_tau = f"{tau}" if tau > 0 else r"\infty"
    cdt.add_current_density_to_plot(main_df, ax, f"$\\tau={label_tau}$")

ax.legend(loc="upper right")
plt.show()