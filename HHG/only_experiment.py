import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

# --- Figure setup ---
fig, ax = plt.subplots()

# --- Experimental data ---
EXP_PATH = "../raw_data_phd/" if os.name == "nt" else "data/"
LASER_DATA = np.loadtxt(EXP_PATH + "HHG/pulse_AB.dat").transpose()
laser_times = LASER_DATA[0]

EXPERIMENTAL_DATA = np.loadtxt(EXP_PATH + "HHG/emitted_signals_in_the_time_domain.dat").transpose()
exp_times = EXPERIMENTAL_DATA[0]
exp_signal = EXPERIMENTAL_DATA[3]

n_exp = len(exp_times)

ax.plot(exp_times, exp_signal / np.max(np.abs(exp_signal)), label="Emitted $E(t)$")
ax.plot(laser_times, LASER_DATA[1] / np.max(np.abs(LASER_DATA[1])), label="Laser $E(t)$", c="k", ls="--")
ax.legend()

ax.set_ylabel("Signal")
ax.set_xlabel("$t$ (ps)")

fig.tight_layout()
fig.subplots_adjust(hspace=0)

plt.show()
