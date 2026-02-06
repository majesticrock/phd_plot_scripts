import numpy as np
import matplotlib.pyplot as plt

import path_appender
path_appender.append()
from get_data import *
from legend import *

main_df = load_panda("HHG", "test", "current_density.json.gz")
times = np.linspace(main_df["t_begin"], main_df["t_end"], main_df["n_measurements"]) / (2 * np.pi)

fig, ax = plt.subplots()

ax.plot(times, main_df["alphas"], label=r"$|\alpha|^2$")
ax.plot(times, main_df["betas"], label=r"$|\beta|^2$")
ax.plot(times, main_df["alphas"] - main_df["betas"], label=r"$\rho(t)$", ls="--")

#ax.plot(main_df["time_greens_real"], label=r"$\Re \alpha$")
#ax.plot(main_df["time_greens_imag"], label=r"$\Im \alpha$")

ax.set_xlabel(legend(r"t / (2 \pi T_L)"))
ax.set_ylabel("Occupations")

fig.tight_layout()
plt.show()