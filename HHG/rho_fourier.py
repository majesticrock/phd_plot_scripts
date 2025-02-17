import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

from scipy.fft import rfftfreq

main_df = load_panda("HHG", "test", "current_density.json.gz")

sample_spacing = (main_df["t_end"] - main_df["t_begin"]) / (main_df["n_measurements"])
frequencies = 2 * np.pi * rfftfreq(main_df["n_measurements"] + 1, sample_spacing)

fig, ax = plt.subplots()
for i in range(33):
    ax.axvline(i, ls="--", color="grey", linewidth=1, alpha=0.5)
    
#ax.plot(frequencies, main_df["current_density_frequency_imag"], label=r"$\Im [\hat{\rho}(\omega)]$")
#ax.plot(frequencies, main_df["current_density_frequency_real"], label=r"$\Re [\hat{\rho}(\omega)]$")
ax.plot(frequencies, np.sqrt(main_df["current_density_frequency_real"]** 2 + main_df["current_density_frequency_imag"]**2))

ax.set_yscale("log")

ax.set_xlabel(legend(r"\omega / \omega_L"))
ax.set_ylabel(legend(r"j(\omega)"))

fig.tight_layout()
plt.show()