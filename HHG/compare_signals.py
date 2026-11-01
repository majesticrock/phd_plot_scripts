import numpy as np
import matplotlib.pyplot as plt


from mrock.get_data import *
data_loader = DataLoader()
from mrock_centralized_scripts.legend import  *

import current_density_fourier as cdf

fig, ax = cdf.create_frame()
MAX_FREQ = 15.5

for i, tau in enumerate([5.0]):
    main_df = data_loader.load_panda("HHG", "cascade/exp_laser/PiFlux", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=2e6, band_width=600, field_amplitude=1, photon_energy=1, tau_diag=tau, t0=0, tau_offdiag=-1))
    label_tau = f"{tau}" if tau > 0 else r"\infty"
    cdf.add_current_density_to_plot(main_df, ax, f"$\\tau={label_tau}$", shift=10**i, max_freq=MAX_FREQ)
    
    main_df = data_loader.load_panda("HHG", "cascade2/exp_laser/PiFlux", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=2e6, band_width=600, field_amplitude=1, photon_energy=1, tau_diag=tau, t0=0, tau_offdiag=-1))
    label_tau = f"{tau}" if tau > 0 else r"\infty"
    cdf.add_current_density_to_plot(main_df, ax, f"$\\tau={label_tau}$", shift=10**i, max_freq=MAX_FREQ)


cdf.add_verticals(main_df["frequencies"], ax, max_freq=MAX_FREQ, positions='odd')

ax.legend(loc="upper right")
plt.show()