import numpy as np
import matplotlib.pyplot as plt


from mrock.get_data import *
data_loader = DataLoader()
from mrock_centralized_scripts.legend import  *

import current_density_fourier as cdf

fig, ax = cdf.create_frame()
MAX_FREQ = 15.5

main_df = data_loader.load_panda("HHG", "cascade_2cycle/dcos_laser/PiFlux", "current_density.json.gz", 
                **hhg_params(T=300, 
                             E_F=118, 
                             v_F=2.5e6, 
                             band_width=600, 
                             field_amplitude=1, 
                             photon_energy=1, 
                             tau_diag=5, 
                             t0=0, 
                             tau_offdiag=-1))
cdf.add_current_density_to_plot(main_df, ax, f"2 cycles", max_freq=MAX_FREQ)
    
main_df = data_loader.load_panda("HHG", "cascade_8cycle/dcos_laser/PiFlux", "current_density.json.gz", 
                **hhg_params(T=300, 
                             E_F=118, 
                             v_F=2.5e6, 
                             band_width=600, 
                             field_amplitude=1, 
                             photon_energy=1, 
                             tau_diag=5, 
                             t0=0, 
                             tau_offdiag=-1))
cdf.add_current_density_to_plot(main_df, ax, f"8 cycles", max_freq=MAX_FREQ)
cdf.add_verticals(main_df["frequencies"], ax, max_freq=MAX_FREQ, positions='odd')

ax.set_ylabel("$S_{A+B}$s")
ax.legend(loc="upper right")
plt.show()