import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

import current_density_fourier as cdf

fig, ax = cdf.create_frame()

#main_df = load_panda("HHG", "nz_100/cosine_laser/PiFlux", "current_density.json.gz", 
#                     **hhg_params(T=0, E_F=0, v_F=1.5e3, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
#cdf.add_current_density_to_plot(main_df, ax, label="$n_z = 100$")
#main_df = load_panda("HHG", "nz_200/cosine_laser/PiFlux", "current_density.json.gz", 
#                     **hhg_params(T=0, E_F=0, v_F=1.5e3, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
#cdf.add_current_density_to_plot(main_df, ax, label="$n_z = 200$")
#main_df = load_panda("HHG", "nz_400/cosine_laser/PiFlux", "current_density.json.gz", 
#                     **hhg_params(T=0, E_F=0, v_F=1.5e3, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
#cdf.add_current_density_to_plot(main_df, ax, label="$n_z = 400$")

main_df = load_panda("HHG", "nz_100/cosine_laser/PiFlux", "current_density.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e4, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
cdf.add_current_density_to_plot(main_df, ax, label="Main")
main_df = load_panda("HHG", "test/cosine_laser/PiFlux", "current_density.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e4, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
cdf.add_current_density_to_plot(main_df, ax, label="Alternative")


cdf.add_verticals(main_df["frequencies"],ax)

ax.legend(loc="upper right")
fig.tight_layout()
plt.show()