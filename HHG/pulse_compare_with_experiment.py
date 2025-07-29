import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from scipy.interpolate import interp1d

import current_density_time as cdt
import current_density_fourier as cdf

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

from scipy.fft import rfft, rfftfreq

MAX_FREQ = 20
DIR = "icelake_cl1"
MODEL = "PiFlux"
v_F = 1.5e6
W = 200
T = 300
E_F = 118
TAU_OFFDIAG=-1
TAU_DIAG=10


fig, axes = cdt.create_frame(nrows=3, figsize=(8.4, 8), 
                             ylabel_list=[legend(r"j_A(t)", "a.b."), 
                                          legend(r"j_B(t)", "a.b."), 
                                          legend(r"j_{A+B}(t)", "a.b.")])

# Reload df_A and df_B with current TAU_DIAG
df_A = load_panda("HHG", f"{DIR}/expA_laser/{MODEL}", "current_density.json.gz", 
                  **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W, 
                               field_amplitude=1., photon_energy=1., 
                               tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0))

df_B = load_panda("HHG", f"{DIR}/expB_laser/{MODEL}", "current_density.json.gz", 
                  **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W, 
                               field_amplitude=1., photon_energy=1., 
                               tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0))
times = np.linspace(0, df_A["t_end"] - df_A["t_begin"], len(df_A["current_density_time"])) / (2 * np.pi)

N = len(times) // 40
signal_A = np.convolve(-df_A["current_density_time"], np.ones(N)/N, mode='same')
signal_B = np.convolve(-df_B["current_density_time"], np.ones(N)/N, mode='same')
inter_A = interp1d(times, signal_A, fill_value=0.0, bounds_error=False)
inter_B = interp1d(times, signal_B, fill_value=0.0, bounds_error=False)

def combined_inter(t, t0):
    if t0 >= 0:
        A = inter_A(t)
        B = inter_B(t - t0)
    else:
        A = inter_A(t + t0)
        B = inter_B(t)
    return A + B

main_df = load_panda("HHG", f"{DIR}/exp_laser/{MODEL}", "current_density.json.gz", 
                     **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W, 
                                  field_amplitude=1., photon_energy=1., 
                                  tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0))

t0_unitless = 0 * main_df["photon_energy"] / (2*np.pi * 0.6582119569509065)
plot_data_combined = combined_inter(times, t0_unitless)

# Time domain plots
axes[0].plot(times, signal_A / np.max(signal_A), label="Simulation")
axes[1].plot(times, signal_B / np.max(signal_B), label="Simulation")
axes[2].plot(times, plot_data_combined / np.max(plot_data_combined), label="Simulation")

EXPERIMENTAL_DATA = np.loadtxt("../raw_data_phd//HHG/emitted_signals_in_the_time_domain.dat").transpose()
exp_times = (15 * 0.03318960199004975 + EXPERIMENTAL_DATA[0]) * main_df["photon_energy"] / (2*np.pi * 0.6582119569509065)
exp_signals = np.array([EXPERIMENTAL_DATA[3], EXPERIMENTAL_DATA[2], EXPERIMENTAL_DATA[1]])

for i, ax in enumerate(axes):
    ax.plot(exp_times, exp_signals[i] / np.max(exp_signals[i]), label="Experiment", linestyle='--')

ax.legend()
plt.show()
