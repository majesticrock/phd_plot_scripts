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

ax.plot(frequencies, -fftshift(main_df["fourier_greens_imag"]), label=r"$\Im [G(\omega)]$")
#ax.plot(frequencies, fftshift(main_df["fourier_greens_real"]), label=r"$\Re [G(\omega)]$", ls="--")

#N = main_df["n_measurements"] // 2 - measurements_per_cycle
#g_time = np.exp(2.0j * np.pi * np.linspace(0., 100., N, endpoint=False)) #main_df["time_greens_real"] + 1j *main_df["time_greens_imag"]
#frequencies = fftshift(fftfreq(N, 100 / N))
#scipy_fft = fftshift(fft(g_time))
#ax.plot(frequencies, 1 / N * scipy_fft.imag, label="scipy imag")
#ax.plot(frequencies, 1 / N * scipy_fft.real, label="scipy real", ls="--")
ax.set_yscale("log")

ax.legend()
ax.set_xlabel(legend(r"\omega / \omega_L"))
ax.set_ylabel(legend(r"G(\omega)"))

fig.tight_layout()
plt.show()