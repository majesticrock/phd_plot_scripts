import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

import current_density_time as cdt
import current_density_fourier as cdf

MAX_FREQ = 20
DIR = "local"
MODEL = "PiFlux"
v_F = 1.5e6
W = 200
T = 300
E_F = 118
TAU_OFFDIAG = -1
TAU_DIAG = 15

sigma = 50e-3
gamma = 50e-3

# Load split currents (dirac and non-dirac)
main_df = load_panda("HHG", f"{DIR}/exp_laser/{MODEL}", "split_current.json.gz", 
                     **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W, 
                                  field_amplitude=1., photon_energy=1., 
                                  tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0))

df_A = load_panda("HHG", f"{DIR}/expA_laser/{MODEL}", "split_current.json.gz", 
                  **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W, 
                               field_amplitude=1., photon_energy=1., 
                               tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0))

df_B = load_panda("HHG", f"{DIR}/expB_laser/{MODEL}", "split_current.json.gz", 
                  **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W, 
                               field_amplitude=1., photon_energy=1., 
                               tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0))

exp_laser_t_max = 15.3335961194029835
N_times = int(main_df["N"] + 1)
times = np.linspace(0, main_df["t_end"] - main_df["t_begin"], N_times) * exp_laser_t_max / main_df["t_end"]
dt = times[1] - times[0]

def cauchy(x, mu, gamma):
    return (1. / np.pi ) * (gamma / ((x - mu)**2 + gamma**2))

kernel = cauchy(times, times[N_times//2], gamma)

# Convolve and normalize Dirac/non-Dirac signals for A, B, combined
def process_signal(df, key):
    signal = np.convolve(np.gradient(df[key]), kernel, mode='same')
    return signal

dirac_main = process_signal(main_df, "dirac_current")
norm = np.max(np.abs(dirac_main))

dirac_main /= norm
non_dirac_main = process_signal(main_df, "non_dirac_current") / norm
dirac_A = process_signal(df_A, "dirac_current") / norm
non_dirac_A = process_signal(df_A, "non_dirac_current") / norm
dirac_B = process_signal(df_B, "dirac_current") / norm
non_dirac_B = process_signal(df_B, "non_dirac_current") / norm
dirac_combined = dirac_A + dirac_B
non_dirac_combined = non_dirac_A + non_dirac_B


# Interpolators for shifting
inter_dirac_A = interp1d(times, dirac_A, fill_value=0.0, bounds_error=False)
inter_non_dirac_A = interp1d(times, non_dirac_A, fill_value=0.0, bounds_error=False)
inter_dirac_B = interp1d(times, dirac_B, fill_value=0.0, bounds_error=False)
inter_non_dirac_B = interp1d(times, non_dirac_B, fill_value=0.0, bounds_error=False)

def combined_inter(t, t0, inter_A, inter_B):
    if t0 >= 0:
        A = inter_A(t)
        B = inter_B(t - t0)
    else:
        A = inter_A(t + t0)
        B = inter_B(t)
    return A + B

t0 = 0
dirac_combined_shifted = combined_inter(times, t0, inter_dirac_A, inter_dirac_B)
non_dirac_combined_shifted = combined_inter(times, t0, inter_non_dirac_A, inter_non_dirac_B)

# Plotting
fig, axes = cdt.create_frame(nrows=3, figsize=(8.4, 10), 
                             ylabel_list=[legend(r"j_\mathrm{lin}(t)", "a.b."), 
                                          legend(r"j_\mathrm{sim}(t)", "a.b."), 
                                          legend(r"j_\mathrm{sim}(t) - j_\mathrm{lin}(t)", "a.b.")])
axes[0].set_xlim(1.7, 15.5)

fig_fft, axes_fft = cdf.create_frame(nrows=3, figsize=(8.4, 10), 
                             ylabel_list=[legend(r"j_\mathrm{lin}(t)", "a.b."), 
                                          legend(r"j_\mathrm{sim}(t)", "a.b."), 
                                          legend(r"j_\mathrm{sim}(t) - j_\mathrm{lin}(t)", "a.b.")])
for ax in axes_fft:
    ax.set_yscale('log')
    ax.set_xlim(0, MAX_FREQ)

# Plot Dirac and non-Dirac contributions in time domain
axes[0].plot(times, dirac_combined_shifted, label="Dirac")
axes[0].plot(times, non_dirac_combined_shifted, label="Non-Dirac", ls="--")
axes[1].plot(times, dirac_main, label="Dirac main")
axes[1].plot(times, non_dirac_main, label="Non-Dirac main", ls="--")
axes[2].plot(times, dirac_main - dirac_combined_shifted, label="Dirac diff")
axes[2].plot(times, non_dirac_main - non_dirac_combined_shifted, label="Non-Dirac diff", ls="--")
axes[2].plot(times, dirac_main + non_dirac_main - dirac_combined_shifted - non_dirac_combined_shifted, label="Total diff", color="k", ls="-.")

# Frequency domain (FFT)
n = len(times) * 4

freqs_scipy = rfftfreq(n, dt)

fft_norm = np.max(np.abs(rfft(dirac_main))**2)

def fft_plot(signal):
    return np.abs(rfft(signal, n))**2 / fft_norm

axes_fft[0].plot(freqs_scipy, fft_plot(dirac_combined_shifted), label="Dirac")
axes_fft[0].plot(freqs_scipy, fft_plot(non_dirac_combined_shifted), label="Non-Dirac", ls="--")
axes_fft[1].plot(freqs_scipy, fft_plot(dirac_main), label="Dirac main")
axes_fft[1].plot(freqs_scipy, fft_plot(non_dirac_main), label="Non-Dirac main", ls="--")
axes_fft[2].plot(freqs_scipy, fft_plot(dirac_main - dirac_combined_shifted), label="Dirac diff")
axes_fft[2].plot(freqs_scipy, fft_plot(non_dirac_main - non_dirac_combined_shifted), label="Non-Dirac diff", ls="--")
axes_fft[2].plot(freqs_scipy, fft_plot(dirac_main + non_dirac_main - dirac_combined_shifted - non_dirac_combined_shifted), label="Total diff", color="k", ls="-.")

for ax in axes_fft:
    ax.set_ylim(1e-9, 5)
    for j in range(0, MAX_FREQ, 2):
        ax.axvline(main_df["photon_energy"] / (2*np.pi*0.6582119569509065) * (j+1), c="k", ls=":", alpha=0.5)

axes[0].legend()
axes_fft[0].legend(loc="upper right")
axes[-1].set_xlabel(legend(r"t", "ps"))
axes_fft[-1].set_xlabel(legend(r"\omega", "THz"))
fig.tight_layout()
fig_fft.tight_layout()
fig.subplots_adjust(hspace=0)
fig_fft.subplots_adjust(hspace=0)
plt.show()