import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

fig, ax = plt.subplots()

def plot_data(df, label, **kwargs):
    frequencies = df["frequencies"]

    for i in range(1, int(np.max(frequencies)) + 1, 2):
        ax.axvline(i, ls="--", color="grey", linewidth=1, alpha=0.5)
    
    current_density = frequencies * (df["current_density_frequency_real"] + 1.0j * df["current_density_frequency_imag"])
    current_density += 1.0j * df["current_density_time"][-1] * np.exp(1.0j * df["t_end"] * frequencies)
    y_data = np.abs(current_density)
    ax.plot(frequencies, y_data / np.max(y_data), label=label, **kwargs)


#main_df = load_panda("HHG", "decay/cosine_laser", "current_density.json.gz", 
#                     **hhg_params(T=0, E_F=0, v_F=1.5e4, band_width=20, field_amplitude=1.6, photon_energy=5.25, decay_time=30))
#plot_data(main_df, r"$W=20, \tau=30$")
#main_df = load_panda("HHG", "decay/cosine_laser", "current_density.json.gz", 
#                     **hhg_params(T=0, E_F=0, v_F=1.5e4, band_width=40, field_amplitude=1.6, photon_energy=5.25, decay_time=30))
#plot_data(main_df, r"$W=40, \tau=30$", ls="--")
#main_df = load_panda("HHG", "decay/cosine_laser", "current_density.json.gz", 
#                     **hhg_params(T=0, E_F=0, v_F=1.5e4, band_width=40, field_amplitude=1.6, photon_energy=5.25, decay_time=10))
#plot_data(main_df, r"$W=40, \tau=10$", ls="--")

#main_df = load_panda("HHG", "decay_4_cycle/cosine_laser", "current_density.json.gz", 
#                     **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=20, field_amplitude=1.6, photon_energy=5.25, decay_time=10))
#plot_data(main_df, r"Decay $W=20$")
main_df = load_panda("HHG", "decay_4_cycle/cosine_laser", "current_density.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=100, field_amplitude=1.6, photon_energy=5.25, decay_time=10))
plot_data(main_df, r"Decay $W=100$")
main_df = load_panda("HHG", "no_decay_4_cycle/cosine_laser", "current_density.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=100, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
plot_data(main_df, r"No decay $W=100$")

ax.set_yscale("log")
ax.set_xlabel(legend(r"\omega / \omega_L"))
ax.set_ylabel(legend(r"\omega j(\omega)", "normalized"))
ax.legend(loc="upper right")
fig.tight_layout()
plt.show()