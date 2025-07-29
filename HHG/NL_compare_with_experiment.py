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
W = 400
T = 300
E_F = 118
TAU_OFFDIAG=-1
TAU_DIAG=30


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

times = np.linspace(0, df_A["t_end"] - df_A["t_begin"], len(df_A["current_density_time"])) / (2 * np.pi)

N = int(0.007 * len(times))
print(N, len(times))
signal_A = np.convolve(df_A["current_density_time"], np.ones(N)/N, mode='same')
signal_B = np.convolve(df_B["current_density_time"], np.ones(N)/N, mode='same')
signal_AB = np.convolve(main_df["current_density_time"], np.ones(N)/N, mode='same')

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

t0_unitless = 0 * main_df["photon_energy"] / (2*np.pi * 0.6582119569509065)
plot_data_combined = combined_inter(times, t0_unitless)

label = f"$\\tau_\\mathrm{{diag}} = {TAU_DIAG}$ ps"
# Time domain plots
axes[0].plot(times, plot_data_combined / np.max(plot_data_combined), label=label)
axes[1].plot(times, signal_AB / np.max(signal_AB), label="Simulation")
axes[2].plot(times, (signal_AB - plot_data_combined) / np.max(signal_AB - plot_data_combined), label="Difference")
# Frequency domain (FFT)
n = len(times) * 4
dt = times[1] - times[0]
freqs_scipy = rfftfreq(n, dt)

fftplot = np.abs(rfft(plot_data_combined, n))**2
axes_fft[0].plot(freqs_scipy, fftplot / np.max(fftplot), label=label)

fftplot = np.abs(rfft(signal_AB, n))**2
axes_fft[1].plot(freqs_scipy, fftplot / np.max(fftplot))

fftplot = np.abs(rfft(signal_AB - plot_data_combined, n))**2
axes_fft[2].plot(freqs_scipy, fftplot / np.max(fftplot))


cdf.add_verticals(freqs_scipy, axes_fft[0], max_freq=MAX_FREQ)
cdf.add_verticals(freqs_scipy, axes_fft[1], max_freq=MAX_FREQ)
cdf.add_verticals(freqs_scipy, axes_fft[2], max_freq=MAX_FREQ)


EXPERIMENTAL_DATA = np.loadtxt("../raw_data_phd//HHG/emitted_signals_in_the_time_domain.dat").transpose()
exp_times = (14 * 0.03318960199004975 + EXPERIMENTAL_DATA[0]) * main_df["photon_energy"] / (2*np.pi * 0.6582119569509065)
exp_signals = np.array([EXPERIMENTAL_DATA[2] + EXPERIMENTAL_DATA[3], EXPERIMENTAL_DATA[1], EXPERIMENTAL_DATA[4]])

n_exp = len(exp_times)
exp_freqs = rfftfreq(n_exp, exp_times[1] - exp_times[0])

for i in range(3):
    axes[i].plot(exp_times, -exp_signals[i] / np.max(exp_signals[i]), label="Experimental data")
    
    exp_fft = np.abs(rfft(exp_signals[i], n_exp))**2
    axes_fft[i].plot(exp_freqs, exp_fft / np.max(exp_fft), label="Experimental data")

# Final plot adjustments
axes[0].legend()
axes_fft[0].legend(loc="upper right")
fig.tight_layout()
fig_fft.tight_layout()

fig.subplots_adjust(hspace=0)
fig_fft.subplots_adjust(hspace=0)

plt.show()
