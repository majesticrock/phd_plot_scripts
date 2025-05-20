import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import legend

fig, ax = plt.subplots()

__PLOT__E___FIELD__ = True

main_df = load_panda("HHG", "cl1_4_cycle/cosine_laser/PiFlux", "current_density.json.gz", 
                     **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
times = np.linspace(0, main_df["t_end"] - main_df["t_begin"], main_df["n_measurements"]) / (2 * np.pi)
A = main_df["laser_function"]

if __PLOT__E___FIELD__:
    dt = (times[1] - times[0])
    E = -np.gradient(A, dt)
    ax.plot(times, E, label="symm.")
else:
    ax.plot(times, A, label="symm.")

main_df = load_panda("HHG", "cl1_4_cycle_shift/cosine_laser/PiFlux", "current_density.json.gz", 
                     **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
times = np.linspace(0, main_df["t_end"] - main_df["t_begin"], main_df["n_measurements"]) / (2 * np.pi)
A = main_df["laser_function"]

if __PLOT__E___FIELD__:
    dt = (times[1] - times[0])
    E = -np.gradient(A, dt)
    ax.plot(times, E, label="asymm.")
else:
    ax.plot(times, A, label="asymm.")

ax.set_xlabel(legend(r"t / (2 \pi T_L)"))
ax.set_ylabel(f"${'E(t)' if __PLOT__E___FIELD__ else 'A(t)'} $ arb. units")
ax.legend()

fig.tight_layout()
plt.show()