import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import current_density_time as cdt
import current_density_fourier as cdf

import path_appender
path_appender.append()
from get_data import *
from legend import *

from scipy.fft import rfft, rfftfreq

MAX_FREQ = 20
DIR = "cascade"
MODEL = "PiFlux"
v_F = 1.5e6
W = 450
T = 300
E_F = 118
TAU_OFFDIAG=-1
TAU_DIAG=5

gamma = 25e-3

FFT_CUTS = [1.7, 6.5]

fig, axes = cdt.create_frame(nrows=3, figsize=(8.4, 10), 
                             ylabel_list=[legend(r"j_\mathrm{lin}(t)", "a.b."), 
                                          legend(r"j_\mathrm{sim}(t)", "a.b."), 
                                          legend(r"j_\mathrm{sim}(t) - j_\mathrm{lin}(t)", "a.b.")])
axes[0].set_xlim(1.7, 6.5)

fig_fft, axes_fft = cdf.create_frame(nrows=3, figsize=(8.4, 10), 
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

exp_laser_t_max =  15.3335961194029835 * 1.5 / 2.#15.3335961194029835 if not new_t else
times = np.linspace(0, df_A["t_end"] - df_A["t_begin"], len(df_A["current_density_time"])) * exp_laser_t_max / df_A["t_end"]

def gaussian(x, mu):
    return (1 / (gamma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * gamma**2))
def cauchy(x, mu):
    return (1. / np.pi ) * (gamma / ((x - mu)**2 + gamma**2))
def laplace(x, mu):
    return np.log(2.) / gamma * np.exp(- 2*np.log(2) / gamma * np.abs(x-mu))
def sech_distrubution(x, mu):
    return (1. / (2. * gamma)) / np.cosh(0.5 * np.pi * (x - mu) / gamma)

N_ave = int(gamma  / (times[1] - times[0]))
__theta_kernel = np.ones(N_ave) / N_ave
__kernel = gaussian(times, times[len(times)//2])

signal_A = -np.convolve(np.gradient(df_A["current_density_time"]), __kernel, mode='same')
signal_B = -np.convolve(np.gradient(df_B["current_density_time"]), __kernel, mode='same')
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

simulation_data = -np.convolve(np.gradient(main_df["current_density_time"]), __kernel, mode='same')

axes[0].plot(times, plot_data_combined / np.max(plot_data_combined), label="Simulation")
axes[1].plot(times, simulation_data / np.max(simulation_data), label="Simulation")
axes[2].plot(times, (simulation_data - plot_data_combined) / np.max(simulation_data - plot_data_combined), label="Simulation")

for ax in axes:
    ax.axvline(FFT_CUTS[0], color='k', linestyle=':')
    ax.axvline(FFT_CUTS[1], color='k', linestyle=':')

# Frequency domain (FFT)
fft_mask = (times >= FFT_CUTS[0]) & (times <= FFT_CUTS[1])

n = len(times[fft_mask]) * 4
dt = times[1] - times[0]
freqs_scipy = rfftfreq(n, dt)

fftplot = np.abs(rfft(plot_data_combined[fft_mask], n))**2
axes_fft[0].plot(freqs_scipy, fftplot / np.max(fftplot))

fftplot = np.abs(rfft(simulation_data[fft_mask], n))**2
axes_fft[1].plot(freqs_scipy, fftplot / np.max(fftplot))

fftplot = np.abs(rfft(simulation_data[fft_mask] - plot_data_combined[fft_mask], n))**2
axes_fft[2].plot(freqs_scipy, fftplot / np.max(fftplot))

EXP_PATH = "../raw_data_phd/" if os.name == "nt" else "data/"
EXPERIMENTAL_DATA = np.loadtxt(EXP_PATH + "HHG/emitted_signals_in_the_time_domain.dat").transpose()
exp_times = (15 * 0.03318960199004975 + EXPERIMENTAL_DATA[0])
exp_signals = np.array([EXPERIMENTAL_DATA[2] + EXPERIMENTAL_DATA[3], EXPERIMENTAL_DATA[1], EXPERIMENTAL_DATA[4]])

n_exp = len(exp_times)
exp_freqs = rfftfreq(n_exp, exp_times[1] - exp_times[0])
exp_fft_mask = (exp_times >= FFT_CUTS[0]) & (exp_times <= FFT_CUTS[1])

LASER_DATA = np.loadtxt(EXP_PATH + "HHG/pulse_AB.dat").transpose()
laser_times = 15 * 0.03318960199004975 + LASER_DATA[0]
laser_plot = -(LASER_DATA[1] + LASER_DATA[2])
laser_deriv = -np.gradient(laser_plot)

for i in range(3):
    axes[i].plot(exp_times, -exp_signals[i] / np.max(exp_signals[i]), label="Experimental data")
    #axes[i].plot(laser_times, laser_plot / np.max(laser_plot), "k:", label="$-E(t)$")
    #axes[i].plot(laser_times, laser_deriv / np.max(laser_deriv), "k--", label="$\\partial_t E(t)$")
    
    exp_fft = np.abs(rfft(exp_signals[i][exp_fft_mask], n_exp))**2
    axes_fft[i].plot(exp_freqs, exp_fft / np.max(exp_fft), label="Experimental data")
    axes_fft[i].set_ylim(1e-9, 5)
    for j in range(0, MAX_FREQ, 2):
        axes_fft[i].axvline(main_df["photon_energy"] / (2*np.pi*0.6582119569509065) * (j+1), c="k", ls=":", alpha=0.5)

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
