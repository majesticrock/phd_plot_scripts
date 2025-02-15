import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import legend

main_df = load_panda("HHG", "test", "test_data.json.gz")
times = np.linspace(main_df["t_begin"], main_df["t_end"], main_df["n_measurements"] + 1) / (2 * np.pi)

fig, ax = plt.subplots()
ax.plot(times, main_df["laser_function"])

ax.set_xlabel(legend(r"t / (2 \pi T_L)"))
ax.set_ylabel("Laser function")

fig.tight_layout()
plt.show()