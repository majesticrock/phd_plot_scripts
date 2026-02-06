import numpy as np
import matplotlib.pyplot as plt

import mrock_centralized_scripts.path_appender as ap
ap.append()
from get_data import *
from legend import *

import os
EXP_PATH = "../raw_data_phd/" if os.name == "nt" else "data/"
EXPERIMENTAL_DATA = np.loadtxt(f"{EXP_PATH}HHG/emitted_signals_in_the_time_domain.dat").transpose()
exp_signals = np.array([EXPERIMENTAL_DATA[1], EXPERIMENTAL_DATA[3], EXPERIMENTAL_DATA[2]])  # A+B, A, B

LASER = np.loadtxt(f"{EXP_PATH}HHG/pulse_AB.dat").transpose()
time_shift = EXPERIMENTAL_DATA[0][np.argmax((exp_signals[0]))] - LASER[0][np.argmax((LASER[1] + LASER[2]))]
print(time_shift)

fig, axes = plt.subplots(nrows=3, figsize=(6, 8), sharex=True)
axes[0].plot(-time_shift + EXPERIMENTAL_DATA[0], exp_signals[0] / np.max(np.abs(exp_signals[0])), label="Exp A+B")
axes[0].plot(LASER[0], (LASER[1] + LASER[2]) / np.max(np.abs(LASER[1] + LASER[2])), label="Laser", ls=":")

axes[1].plot(-time_shift + EXPERIMENTAL_DATA[0], exp_signals[1] / np.max(np.abs(exp_signals[1])), label="Exp A")
axes[1].plot(LASER[0], LASER[1] / np.max(np.abs(LASER[1])), label="Laser", ls=":")

axes[2].plot(-time_shift + EXPERIMENTAL_DATA[0], exp_signals[2] / np.max(np.abs(exp_signals[2])), label="Exp B")
axes[2].plot(LASER[0], LASER[2] / np.max(np.abs(LASER[2])), label="Laser", ls=":")

axes[-1].set_xlabel(r"$t$ (ps)")
axes[0].set_ylabel(legend(r"\partial_t j(t)", "A+B"))
axes[1].set_ylabel(legend(r"\partial_t j(t)", "A"))
axes[2].set_ylabel(legend(r"\partial_t j(t)", "B"))
fig.tight_layout()



exp_signals = -np.cumsum(np.array([EXPERIMENTAL_DATA[1], EXPERIMENTAL_DATA[3], EXPERIMENTAL_DATA[2]]), axis=1)  # A+B, A, B
time_shift = EXPERIMENTAL_DATA[0][np.argmax((exp_signals[0]))] - LASER[0][np.argmax((LASER[1] + LASER[2]))]
print(time_shift)

fig_diff, axes_diff = plt.subplots(nrows=3, figsize=(6, 8), sharex=True)
axes_diff[0].plot(-time_shift + EXPERIMENTAL_DATA[0], exp_signals[0] / np.max(np.abs(exp_signals[0])), label="Exp A+B")
axes_diff[0].plot(LASER[0], (LASER[1] + LASER[2]) / np.max(np.abs(LASER[1] + LASER[2])), label="Laser", ls=":")

axes_diff[1].plot(-time_shift + EXPERIMENTAL_DATA[0], exp_signals[1] / np.max(np.abs(exp_signals[1])), label="Exp A")
axes_diff[1].plot(LASER[0], LASER[1] / np.max(np.abs(LASER[1])), label="Laser", ls=":")

axes_diff[2].plot(-time_shift + EXPERIMENTAL_DATA[0], exp_signals[2] / np.max(np.abs(exp_signals[2])), label="Exp B")
axes_diff[2].plot(LASER[0], LASER[2] / np.max(np.abs(LASER[2])), label="Laser", ls=":")

axes_diff[-1].set_xlabel(r"$t$ (ps)")
axes_diff[0].set_ylabel(legend(r"j(t)", "A+B"))
axes_diff[1].set_ylabel(legend(r"j(t)", "A"))
axes_diff[2].set_ylabel(legend(r"j(t)", "B"))
fig_diff.tight_layout()

plt.show()