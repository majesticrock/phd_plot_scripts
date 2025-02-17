import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

from scipy.fft import *

main_df = load_panda("HHG", "test", "test_data.json.gz")

#times = np.linspace(main_df["t_begin"], main_df["t_end"], main_df["n_measurements"] + 1)
sample_spacing = (main_df["t_end"] - main_df["t_begin"]) / (main_df["n_measurements"] + 1)
measurements_per_cycle = main_df["n_measurements"] // main_df["n_laser_cycles"]
frequencies = 2 * np.pi * fftshift(fftfreq(main_df["n_measurements"] // 2 - measurements_per_cycle, sample_spacing))

fig, ax = plt.subplots()
phase_shift = 1# np.exp(1.0j * (frequencies / (2 * np.pi)) * 0.5 * (main_df["t_end"] - main_df["t_begin"]))

#ax.plot(frequencies, -fftshift(main_df["fourier_greens_imag"]) * phase_shift, label=r"$\Im [G(\omega)]$")
#ax.plot(frequencies, fftshift(main_df["fourier_greens_real"]), label=r"$\Re [G(\omega)]$", ls="--")

ax.plot(frequencies, fftshift(np.sqrt(main_df["fourier_greens_imag"]**2 + main_df["fourier_greens_real"]**2)) * phase_shift, label=r"$\Im [G(\omega)]$")

N = main_df["n_measurements"] // 2 - measurements_per_cycle
g_time = main_df["time_greens_real"] + 1j *main_df["time_greens_imag"]
scipy_fft = fftshift(fft(g_time))
ax.plot(frequencies, np.abs(scipy_fft), label="scipy imag", ls="--")
#ax.plot(frequencies, 1 / N * scipy_fft.real, label="scipy real", ls="--")
ax.set_yscale("log")

ax.legend()
ax.set_xlabel(legend(r"\omega / \omega_L"))
ax.set_ylabel(legend(r"G(\omega)"))

fig.tight_layout()
plt.show()