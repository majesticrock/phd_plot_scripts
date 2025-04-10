import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

main_df = load_panda("HHG", "no_decay_4_cycle/cosine_laser", "current_density.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e3, band_width=100, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))

frequencies = main_df["frequencies"]

fig, ax = plt.subplots()
for i in range(1, int(np.max(frequencies)) + 1, 2):
    ax.axvline(i, ls="--", color="grey", linewidth=1, alpha=0.5)
    
current_density = frequencies * (main_df["current_density_frequency_real"] + 1.0j * main_df["current_density_frequency_imag"])
current_density += 1.0j * main_df["current_density_time"][-1] * np.exp(1.0j * main_df["t_end"] * frequencies)
y_data = np.abs(current_density)
ax.plot(frequencies, y_data / np.max(y_data))

ax.set_yscale("log")

ax.set_xlabel(legend(r"\omega / \omega_L"))
ax.set_ylabel(legend(r"j(\omega)", "normalized"))

fig.tight_layout()
plt.show()