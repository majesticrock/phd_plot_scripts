import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

def plot_data(df, label, **kwargs):
    times = np.linspace(main_df["t_begin"], main_df["t_end"], len(main_df["current_density_time"])) / (2 * np.pi)
    ax.plot(times, main_df["current_density_time"], label=label, **kwargs)

fig, ax = plt.subplots()

#main_df = load_panda("HHG", "decay/cosine_laser", "current_density.json.gz", 
#                     **hhg_params(T=0, E_F=0, v_F=1.5e4, band_width=20, field_amplitude=1.6, photon_energy=5.25, decay_time=30))
#plot_data(main_df, r"$W=20, \tau=30$")
#
#main_df = load_panda("HHG", "decay/cosine_laser", "current_density.json.gz", 
#                     **hhg_params(T=0, E_F=0, v_F=1.5e4, band_width=40, field_amplitude=1.6, photon_energy=5.25, decay_time=30))
#plot_data(main_df, r"$W=40, \tau=30$", ls="--")
#
#main_df = load_panda("HHG", "decay/cosine_laser", "current_density.json.gz", 
#                     **hhg_params(T=0, E_F=0, v_F=1.5e4, band_width=40, field_amplitude=1.6, photon_energy=5.25, decay_time=10))
#plot_data(main_df, r"$W=40, \tau=10$", ls="-.")
##########
#main_df = load_panda("HHG", "decay_4_cycle/cosine_laser", "current_density.json.gz", 
#                     **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=20, field_amplitude=1.6, photon_energy=5.25, decay_time=10))
#plot_data(main_df, r"2 cycles", ls="-")
#main_df = load_panda("HHG", "test_4_cycle/cosine_laser", "current_density.json.gz", 
#                     **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=20, field_amplitude=1.6, photon_energy=5.25, decay_time=10))
#plot_data(main_df, r"2 cycles", ls="-")
##########

main_df = load_panda("HHG", "no_decay_4_cycle/cosine_laser", "current_density.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=100, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
plot_data(main_df, r"No decay $W=100$")

ax.set_xlabel(legend(r"t / T_\mathrm{L}")) # T_L = 2 pi / omega_L
ax.set_ylabel(legend(r"j(t)"))

ax.legend()

fig.tight_layout()
plt.show()