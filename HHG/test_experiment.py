import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from scipy.interpolate import interp1d

import current_density_time as cdt
import current_density_fourier as cdf

import mrock_centralized_scripts.path_appender as ap
ap.append()
from get_data import *
from legend import *

from scipy.fft import rfft, rfftfreq

MAX_FREQ = 20
BASE_SHIFT = 2.5
TIME_RATIO = 0.03
DIR = "cascade_prec"
MODEL = "PiFlux"
v_F = 1.5e6
W = 200
T = 300
E_F = 118
TAU_OFFDIAG=-1
TAU_DIAG=-1

tau_diag_values = [10, 20, 50]  # Add more values as needed
t0_values = [0]  # You can loop over multiple t0 as well

colors = plt.cm.viridis(np.linspace(0, 1, len(tau_diag_values)))


fig, axes = cdt.create_frame(nrows=3, figsize=(4.4, 8), 
                             ylabel_list=[legend(r"j_\mathrm{lin}(t)", "a.b."), 
                                          legend(r"j_\mathrm{sim}(t)", "a.b."), 
                                          legend(r"j_\mathrm{sim}(t) - j_\mathrm{lin}(t)", "a.b.")])

fig_fft, axes_fft = cdf.create_frame(nrows=3, figsize=(4.4, 8), 
                             ylabel_list=[legend(r"j_\mathrm{lin}(t)", "a.b."), 
                                          legend(r"j_\mathrm{sim}(t)", "a.b."), 
                                          legend(r"j_\mathrm{sim}(t) - j_\mathrm{lin}(t)", "a.b.")])
for ax in axes_fft:
    ax.set_yscale('log')
    ax.set_xlim(0, MAX_FREQ)

for tau_idx, TAU_DIAG in enumerate(tau_diag_values):
    color = colors[tau_idx]
    
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

    for i, t0 in enumerate(t0_values):
        main_df = load_panda("HHG", f"{DIR}/exp_laser/{MODEL}", "current_density.json.gz", 
                             **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W, 
                                          field_amplitude=1., photon_energy=1., 
                                          tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=t0))
        
        times2 = np.linspace(0, main_df["t_end"] - main_df["t_begin"], len(main_df["current_density_time"])) / (2 * np.pi)
        t0_unitless = t0 * main_df["photon_energy"] / (2*np.pi * 0.6582119569509065)
        plot_data_combined = combined_inter(times2, t0_unitless)
        
        label = f"$\\tau_\\mathrm{{diag}} = {TAU_DIAG}$ ps"

        # Time domain plots
        axes[0].plot(times2, plot_data_combined - TIME_RATIO * tau_idx * BASE_SHIFT, label=label, color=color)
        cdt.add_current_density_to_plot(main_df, axes[1], normalize=False, color=color, shift=TIME_RATIO * tau_idx * BASE_SHIFT)
        cdt.add_current_density_to_plot(main_df, axes[2], substract=lambda t: combined_inter(t, t0_unitless), 
                                        label=label, normalize=False, color=color, shift=TIME_RATIO * tau_idx * BASE_SHIFT)

        # Frequency domain (FFT)
        n = len(times2) * 4
        dt = times2[1] - times2[0]
        freqs_scipy = rfftfreq(n, dt)

        fftplot = np.abs(rfft(plot_data_combined, n))*(BASE_SHIFT**(-2*tau_idx))
        axes_fft[0].plot(freqs_scipy, fftplot, label=label, color=color)

        fftplot = np.abs(rfft(main_df["current_density_time"], n))*(BASE_SHIFT**(-2*tau_idx))
        axes_fft[1].plot(freqs_scipy, fftplot, color=color)

        fftplot = np.abs(rfft(main_df["current_density_time"] - plot_data_combined, n))*(BASE_SHIFT**(-2*tau_idx))
        axes_fft[2].plot(freqs_scipy, fftplot, color=color)

        # Optional: Add vertical lines for harmonics
        cdf.add_verticals(freqs_scipy, axes_fft[0], max_freq=MAX_FREQ)
        cdf.add_verticals(freqs_scipy, axes_fft[1], max_freq=MAX_FREQ)
        cdf.add_verticals(freqs_scipy, axes_fft[2], max_freq=MAX_FREQ)

# Final plot adjustments
axes[0].legend()
axes_fft[0].legend(loc="upper right")
fig.tight_layout()
fig_fft.tight_layout()

fig.subplots_adjust(hspace=0)
fig_fft.subplots_adjust(hspace=0)

plt.show()
