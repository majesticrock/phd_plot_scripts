import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))
TIME_TO_UNITLESS = 2 * np.pi * 0.6582119569509065

BAND_WIDTH=300
T_AVE=0.035
MAX_FREQ = 20

DIR = "test2"
main_df = load_panda("HHG", f"{DIR}/exp_laser/PiFlux", "split_current.json.gz", 
                     **hhg_params(T=300, E_F=118, v_F=1.5e6, band_width=BAND_WIDTH, 
                                  field_amplitude=1, photon_energy=1., 
                                  tau_diag=15, tau_offdiag=-1, t0=0))

fig, axes = plt.subplots(nrows=2)
axes[0].set_xlabel(legend(r"t / T_\mathrm{L}"))

N_times =  int(main_df["N"] + 1)
times = np.linspace(0, main_df["t_end"] - main_df["t_begin"], int(main_df["N"] + 1))
times /= (2*np.pi)
dt = times[1] - times[0]

def gaussian(x, mu=0, sigma=1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

N = int( T_AVE * main_df["photon_energy"] / TIME_TO_UNITLESS / (times[1] - times[0]) )
sigma =  T_AVE * main_df["photon_energy"] / TIME_TO_UNITLESS
kernel = gaussian(times, times[N_times//2], sigma) #np.ones(N)/N

#main_df["lowest_energy_current"]#
#main_df["high_energy_current"] #
lowest_data_time = np.convolve(main_df["lowest_energy_current"], kernel, mode='same')
low_dirac_data_time = np.convolve(main_df["low_dirac_current"], kernel, mode='same')
high_dirac_data_time = np.convolve(main_df["high_dirac_current"] , kernel, mode='same')
highest_data_time = np.convolve(main_df["highest_energy_current"] , kernel, mode='same')

axes[0].plot(times, lowest_data_time, label="$E < -E_D$", ls="--")
axes[0].plot(times, low_dirac_data_time, label="$E > -E_D$")
axes[0].plot(times, high_dirac_data_time, label="$E < E_D$")
axes[0].plot(times, highest_data_time, label="$E > E_D$", ls="--")
#axes[0].plot(times, main_df["laser_function"], "k--", label="Laser")
axes[0].legend()

from scipy.fft import rfft, rfftfreq
axes[1].set_xlabel(legend(r"\omega / \omega_\mathrm{L}"))
n = N_times * 4

freqs_scipy = rfftfreq(n, dt)
lowest_data = np.abs(rfft(lowest_data_time, n))
low_dirac_data  = np.abs(rfft(low_dirac_data_time, n))
high_dirac_data = np.abs(rfft(high_dirac_data_time, n))
highest_data = np.abs(rfft(highest_data_time, n))

norm = np.max(high_dirac_data_time)

for i in range(1, MAX_FREQ + 1, 2):
    axes[1].axvline(i, ls="--", c="k", alpha=0.5)

axes[1].plot(freqs_scipy, lowest_data / norm,     label="$E < -E_D$", ls="--")
axes[1].plot(freqs_scipy, low_dirac_data / norm,  label="$E > -E_D$")
axes[1].plot(freqs_scipy, high_dirac_data / norm, label="$E < E_D$")
axes[1].plot(freqs_scipy, highest_data / norm,    label="$E > E_D$", ls="--")
axes[1].set_yscale("log")
axes[1].set_xlim(0, MAX_FREQ + 0.5)

fig.tight_layout()
plt.show()