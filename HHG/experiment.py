import numpy as np
import matplotlib.pyplot as plt

import current_density_fourier as cdf

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

df_A = load_panda("HHG", "test/expA_laser/Honeycomb", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=1e6, band_width=3300, field_amplitude=1., photon_energy=1., decay_time=1000))
df_B = load_panda("HHG", "test/expB_laser/Honeycomb", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=1e6, band_width=3300, field_amplitude=1., photon_energy=1., decay_time=1000))

signal_A = cdf.compute_current_density(df_A)
signal_B = cdf.compute_current_density(df_B)
signal_AB = signal_A + signal_B

fig, axes = cdf.create_frame(nrows=3, figsize=(6.4, 8))
frequencies = df_A["frequencies"]
cdf.add_verticals(frequencies, axes[0], max_freq=40)
cdf.add_verticals(frequencies, axes[1], max_freq=40)
cdf.add_verticals(frequencies, axes[2], max_freq=40)

axes[0].plot(frequencies, np.abs(signal_A), label="FFT A")
axes[0].plot(frequencies, np.abs(signal_B), label="FFT B")
axes[0].plot(frequencies, np.abs(signal_AB), label="FFT A+B")
axes[0].set_xlim(0, 40)
axes[0].legend()

for i, t0 in enumerate([0, 0.5, 1, 1.5]):
    main_df = load_panda("HHG", f"test_{t0}/exp_laser/Honeycomb", "current_density.json.gz", 
                        **hhg_params(T=300, E_F=118, v_F=1e6, band_width=3300, field_amplitude=1., photon_energy=1., decay_time=1000))
    cdf.add_current_density_to_plot(main_df, axes[1], max_freq=40, label=f"Proper $t_0 = {t0}$ ps", shift=10**(-i))
    cdf.add_current_density_to_plot(main_df, axes[2], max_freq=40, substract=signal_AB * np.exp(1.0j * frequencies * t0), label=f"NL $t_0 = {t0}$ ps", shift=10**(-i))

axes[1].legend(loc='upper right')
axes[2].legend(loc='upper right')

fig.tight_layout()
plt.show()