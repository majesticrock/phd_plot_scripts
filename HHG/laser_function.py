import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import legend

fig, ax = plt.subplots()

main_df = load_panda("HHG", "test0/cosine_laser/PiFlux", "current_density.json.gz", 
                     **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
times = np.linspace(0, main_df["t_end"] - main_df["t_begin"], main_df["n_measurements"]) / (2 * np.pi)
ax.plot(times, main_df["laser_function"])

main_df = load_panda("HHG", "test1/cosine_laser/PiFlux", "current_density.json.gz", 
                     **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
times = np.linspace(0, main_df["t_end"] - main_df["t_begin"], main_df["n_measurements"]) / (2 * np.pi)
ax.plot(times, main_df["laser_function"], ls="--")

main_df = load_panda("HHG", "test05/cosine_laser/PiFlux", "current_density.json.gz", 
                     **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
times = np.linspace(0, main_df["t_end"] - main_df["t_begin"], main_df["n_measurements"]) / (2 * np.pi)
ax.plot(times, main_df["laser_function"], ls="-.")

ax.set_xlabel(legend(r"t / (2 \pi T_L)"))
ax.set_ylabel("Laser function")

fig.tight_layout()
plt.show()