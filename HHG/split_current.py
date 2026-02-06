import numpy as np
import matplotlib.pyplot as plt

import mrock_centralized_scripts.path_appender as ap
ap.append()
from get_data import *
from legend import *

FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))
TIME_TO_UNITLESS = 2 * np.pi * 0.6582119569509065

BAND_WIDTH=300
T_AVE=0.05
TAU_DIAG = 15
MAX_FREQ = 20

LASER = "powerlaw1_laser"
DIR = "test"
main_df = load_panda("HHG", f"{DIR}/{LASER}/PiFlux", "split_current.json.gz", 
                     **hhg_params(T=300, E_F=118, v_F=1.5e6, band_width=BAND_WIDTH, 
                                  field_amplitude=1.6, photon_energy=5.25, 
                                  tau_diag=TAU_DIAG, tau_offdiag=-1, t0=8))

def cauchy(x, mu, gamma):
    return (1. / np.pi ) * (gamma / ((x - mu)**2 + gamma**2))

fig, axes = plt.subplots(nrows=2)

N_times =  int(main_df["N"] + 1)
times = np.linspace(0, main_df["t_end"] - main_df["t_begin"], int(main_df["N"] + 1)) / (2 * np.pi)
dt = times[1] - times[0]
sigma =  T_AVE * main_df["photon_energy"] / TIME_TO_UNITLESS
kernel = cauchy(times, times[N_times//2], sigma)

##########################
#full_df = load_panda("HHG", f"cascade_prec/exp_laser/PiFlux", "current_density.json.gz", 
#                     **hhg_params(T=300, E_F=118, v_F=1.5e6, band_width=BAND_WIDTH, 
#                                  field_amplitude=1, photon_energy=1., 
#                                  tau_diag=TAU_DIAG, tau_offdiag=-1, t0=0))
#full_N = int(full_df["N"] + 1)
#full_times = np.linspace(full_df["t_begin"], full_df["t_end"], full_N) / (2 * np.pi)
#full_data_time = np.gradient(np.convolve(full_df["current_density_time"], cauchy(full_times, full_times[full_N//2], sigma), mode='same'))
#full_data_time /= np.max(full_data_time)
#axes[0].plot(full_times, full_data_time, "k", label="Full")
##########################

dirac_data_time     = np.gradient(np.convolve(main_df["dirac_current"], kernel, mode='same'))
non_dirac_data_time = np.gradient(np.convolve(main_df["non_dirac_current"] , kernel, mode='same'))

norm = np.max(dirac_data_time) + np.max(non_dirac_data_time)
dirac_data_time /= norm
non_dirac_data_time /= norm


axes[0].plot(times, dirac_data_time, label="Dirac")
axes[0].plot(times, non_dirac_data_time, label="Non-Dirac", ls="--")
axes[0].plot(times, dirac_data_time + non_dirac_data_time, label="Checksum", ls=":", color="gray", linewidth=3)

axes[0].legend()
axes[0].set_xlabel(legend(r"t / T_\mathrm{L}"))

from scipy.fft import rfft, rfftfreq
axes[1].set_xlabel(legend(r"\omega / \omega_\mathrm{L}"))
n = N_times * 4

freqs_scipy = rfftfreq(n, dt)
dirac_data      = np.abs(rfft((dirac_data_time    ), n))**2
non_dirac_data  = np.abs(rfft((non_dirac_data_time), n))**2

norm = np.max(dirac_data) + np.max(non_dirac_data)
dirac_data /= norm
non_dirac_data /= norm

for i in range(1, MAX_FREQ + 1, 2):
    axes[1].axvline(i, ls="--", c="k", alpha=0.5)

##########################
#full_data = np.abs(rfft((full_data_time), 2 * full_N))**2
#full_data /= np.max(full_data)
#axes[1].plot(rfftfreq(2 * full_N, full_times[1] - full_times[0]), full_data, "k", label="Full")
##########################

axes[1].plot(freqs_scipy, dirac_data)
axes[1].plot(freqs_scipy, non_dirac_data, ls="--")

axes[1].set_yscale("log")
axes[1].set_xlim(0, MAX_FREQ + 0.5)

axes[0].set_ylabel("$\partial_t j(t)$ (arb. units)")
axes[1].set_ylabel("$|\\mathcal{F} [\\partial_t j] (\\omega)|^2$ (arb. units)")

fig.tight_layout()
plt.show()