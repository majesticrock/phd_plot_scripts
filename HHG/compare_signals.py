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


#main_df = load_panda("HHG", "test_no_decay/cosine_laser", "current_density.json.gz", 
#                     **hhg_params(T=0, E_F=0, v_F=1.5e3, band_width=10, field_amplitude=1.6, photon_energy=5.25))
#plot_data(main_df, r"No decay")
#main_df = load_panda("HHG", "test_small_decay/cosine_laser", "current_density.json.gz", 
#                     **hhg_params(T=0, E_F=0, v_F=1.5e3, band_width=10, field_amplitude=1.6, photon_energy=5.25))
#plot_data(main_df, r"Small decay", ls="--")
#main_df = load_panda("HHG", "test_decay/cosine_laser", "current_density.json.gz", 
#                     **hhg_params(T=0, E_F=0, v_F=1.5e3, band_width=10, field_amplitude=1.6, photon_energy=5.25))
#plot_data(main_df, r"Normal decay", ls="-.")

main_df = load_panda("HHG", "test_decay/cosine_laser", "current_density.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e4, band_width=20, field_amplitude=1.6, photon_energy=5.25, decay_time=30))
plot_data(main_df, r"$W=20$")
main_df = load_panda("HHG", "test_decay/cosine_laser", "current_density.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e4, band_width=40, field_amplitude=1.6, photon_energy=5.25, decay_time=30))
plot_data(main_df, r"$W=40$", ls="--")
main_df = load_panda("HHG", "test_decay/cosine_laser", "current_density.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e4, band_width=40, field_amplitude=1.6, photon_energy=5.25, decay_time=10))
plot_data(main_df, r"$W=40, \tau=10$", ls="--")
main_df = load_panda("HHG", "test_decay_2/cosine_laser", "current_density.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e4, band_width=40, field_amplitude=1.6, photon_energy=5.25, decay_time=10))
plot_data(main_df, r"$W=40, \tau=10$ v2", ls="-.")

ax.set_yscale("log")
ax.set_xlabel(legend(r"\omega / \omega_L"))
ax.set_ylabel(legend(r"j(\omega)", "normalized"))
ax.legend(loc="upper right")
fig.tight_layout()
plt.show()