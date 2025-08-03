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
TAU_DIAG=20

FFT_CUTS = [1.7, 6.5]

fig, axes = cdt.create_frame(nrows=3, figsize=(8.4, 8), 
                             ylabel_list=[legend(r"j_\mathrm{lin}(t)", "a.b."), 
                                          legend(r"j_\mathrm{sim}(t)", "a.b."), 
                                          legend(r"j_\mathrm{sim}(t) - j_\mathrm{lin}(t)", "a.b.")])

fig_fft, axes_fft = cdf.create_frame(nrows=3, figsize=(8.4, 8), 
                             ylabel_list=[legend(r"j_\mathrm{lin}(t)", "a.b."), 
                                          legend(r"j_\mathrm{sim}(t)", "a.b."), 
                                          legend(r"j_\mathrm{sim}(t) - j_\mathrm{lin}(t)", "a.b.")])
for ax in axes_fft:
    ax.set_yscale('log')
    ax.set_xlim(0, MAX_FREQ)

# Reload df_A and df_B with current TAU_DIAG
df_A = load_panda("HHG", f"{DIR}/expA_laser/{MODEL}", "current_density.json.gz", 
                  **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W, 
                               field_amplitude=1., photon_energy=1., 
                               tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0))

df_B = load_panda("HHG", f"{DIR}/expB_laser/{MODEL}", "current_density.json.gz", 
                  **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W, 
                               field_amplitude=1., photon_energy=1., 
                               tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0))

main_df = load_panda("HHG", f"{DIR}/exp_laser/{MODEL}", "current_density.json.gz", 
                     **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W, 
                                  field_amplitude=1., photon_energy=1., 
                                  tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0))

exp_laser_t_max = 15.3335961194029835
times  = np.linspace(0, df_A["t_end"]    - df_A["t_begin"],    len(df_A["current_density_time"]))    * exp_laser_t_max / df_A["t_end"]

signal_A = df_A["current_density_time"]
signal_B = df_B["current_density_time"]
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

t0 = 0
plot_data_combined = combined_inter(times, t0)

label = f"$\\tau_\\mathrm{{diag}} = {TAU_DIAG}$ ps"
# Time domain plots
axes[0].plot(times, plot_data_combined / np.max(plot_data_combined), label=label)
axes[1].plot(times, main_df["current_density_time"] / np.max(main_df["current_density_time"]), label="Simulation")
axes[2].plot(times, (main_df["current_density_time"] - plot_data_combined) / np.max(main_df["current_density_time"] - plot_data_combined), label="Difference")

for ax in axes:
    ax.axvline(FFT_CUTS[0], color='k', linestyle=':')
    ax.axvline(FFT_CUTS[1], color='k', linestyle=':')

# Frequency domain (FFT)
fft_mask = (times >= FFT_CUTS[0]) & (times <= FFT_CUTS[1])

n = len(times[fft_mask]) * 4
dt = times[1] - times[0]
freqs_scipy = rfftfreq(n, dt)

fftplot = np.abs(rfft(plot_data_combined[fft_mask], n))
axes_fft[0].plot(freqs_scipy, fftplot / np.max(fftplot), label=label)

fftplot = np.abs(rfft(main_df["current_density_time"][fft_mask], n))
axes_fft[1].plot(freqs_scipy, fftplot / np.max(fftplot))

fftplot = np.abs(rfft(main_df["current_density_time"][fft_mask] - plot_data_combined[fft_mask], n))**2
axes_fft[2].plot(freqs_scipy, fftplot / np.max(fftplot))

EXPERIMENTAL_DATA = np.loadtxt("../raw_data_phd//HHG/emitted_signals_in_the_time_domain.dat").transpose()
exp_times = (15 * 0.03318960199004975 + EXPERIMENTAL_DATA[0])
exp_signals = np.array([EXPERIMENTAL_DATA[2] + EXPERIMENTAL_DATA[3], EXPERIMENTAL_DATA[1], EXPERIMENTAL_DATA[4]])

n_exp = len(exp_times)
exp_freqs = rfftfreq(n_exp, exp_times[1] - exp_times[0])
exp_fft_mask = (exp_times >= FFT_CUTS[0]) & (exp_times <= FFT_CUTS[1])

LASER_DATA = np.loadtxt("../raw_data_phd//HHG/pulse_AB.dat").transpose()
laser_times = 15 * 0.03318960199004975 + LASER_DATA[0]
laser_plot = -(LASER_DATA[1] + LASER_DATA[2])

for i in range(3):
    axes[i].plot(exp_times, -exp_signals[i] / np.max(exp_signals[i]), label="Experimental data")
    #axes[i].plot(laser_times, laser_plot / np.max(laser_plot), label="Laser pulse", linestyle='--')
    
    exp_fft = np.abs(rfft(exp_signals[i][exp_fft_mask], n_exp))**2
    axes_fft[i].plot(exp_freqs, exp_fft / np.max(exp_fft), label="Experimental data")


# Final plot adjustments
axes[0].legend()
axes_fft[0].legend(loc="upper right")
axes[-1].set_xlabel(legend(r"t", "ps"))
axes_fft[-1].set_xlabel(legend(r"\omega", "THz"))
fig.tight_layout()
fig_fft.tight_layout()

fig.subplots_adjust(hspace=0)
fig_fft.subplots_adjust(hspace=0)

plt.show()
