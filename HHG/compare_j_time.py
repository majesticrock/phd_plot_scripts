import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

import current_density_time as cdt
import laser_function as lf

fig, ax = cdt.create_frame()

for i, tau in enumerate([30, 1000]):
    main_df = load_panda("HHG", "test/quench_laser/PiFlux", "current_density.json.gz", 
                     **hhg_params(T=300, 
                                  E_F=118, 
                                  v_F=1.5e5, 
                                  band_width=400, 
                                  field_amplitude=1.6, 
                                  photon_energy=5.25, 
                                  tau_diag=tau,
                                  tau_offdiag=30,
                                  t0=16))
    label_tau = f"{tau}" if tau > 0 else r"\infty"
    cdt.add_current_density_to_plot(main_df, ax, f"$\\tau_{{\mathrm{{diag}}}}={label_tau},\\tau_{{\mathrm{{offdiag}}}}=30$")
    
    main_df = load_panda("HHG", "test/quench_laser/PiFlux", "current_density.json.gz", 
                     **hhg_params(T=300, 
                                  E_F=118, 
                                  v_F=1.5e5, 
                                  band_width=400, 
                                  field_amplitude=1.6, 
                                  photon_energy=5.25, 
                                  tau_diag=tau,
                                  tau_offdiag=1000,
                                  t0=16))
    cdt.add_current_density_to_plot(main_df, ax, f"$\\tau_{{\mathrm{{diag}}}}={label_tau},\\tau_{{\mathrm{{offdiag}}}}=1000$")

lf.add_laser_to_plot(main_df, ax, color='k', ls="--", label="$A(t)$", normalize=True)

ax.legend(loc="upper left")
plt.show()