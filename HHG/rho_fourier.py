import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

from scipy.fft import rfft, rfftfreq

main_df = load_panda("HHG", "test", "test_data.json.gz")

#times = np.linspace(main_df["t_begin"], main_df["t_end"], main_df["n_measurements"] + 1)
sample_spacing = (main_df["t_end"] - main_df["t_begin"]) / (main_df["n_measurements"] + 1)

frequencies = 2 * np.pi * rfftfreq(main_df["n_measurements"] + 1, sample_spacing)

fig, ax = plt.subplots()

ax.plot(frequencies, main_df["fourier_rho_imag"], label=r"$\Im [\hat{\rho}(\omega)]$")
ax.plot(frequencies, main_df["fourier_rho_real"], label=r"$\Re [\hat{\rho}(\omega)]$")


y = rfft((main_df["alphas"] - main_df["betas"]))

ax.plot(frequencies, np.real(y), label=r"$\Im [\mathrm{scipy}]$", ls="--")
ax.plot(frequencies, np.imag(y), label=r"$\Re [\mathrm{scipy}]$", ls="--")

ax.set_xlabel(legend(r"\omega / \omega_L"))
ax.set_ylabel(legend(r"\hat{\rho}(\omega)"))

fig.tight_layout()
plt.show()