import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq

import current_density_time as cdt
import current_density_fourier as cdf

import mrock_centralized_scripts.path_appender as ap
ap.append()
from get_data import *
from legend import *

MAX_FREQ = 15
TIME_TO_UNITLESS = 2 * np.pi * 0.6582119569509065
print(TIME_TO_UNITLESS)
FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))

# FFT window (unitless time)
FFT_TMIN = 3.0
FFT_TMAX = 7.0

# Default parameters in one place
params = {
    "DIR": "cascade_16",
    "MODEL": "PiFlux",
    "v_F": 2e6,
    "W": 600,
    "T": 300,
    "E_F": 118,
    "TAU_OFFDIAG": -1,
    "TAU_DIAG": 5,
    "T_AVE": 28
}


def gaussian(x, mu, gamma):
    return (1 / (gamma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * gamma**2))
def cauchy(x, mu, gamma):
    return (1. / np.pi ) * (gamma / ((x - mu)**2 + gamma**2))
def laplace(x, mu, gamma):
    return np.log(2.) / gamma * np.exp(- 2*np.log(2) / gamma * np.abs(x-mu))
def sech_distrubution(x, mu, gamma):
    return (1. / (2. * gamma)) / np.cosh(0.5 * np.pi * (x - mu) / gamma)
def cos_dist(N):
    return (1. - np.cos(np.pi * np.linspace(0., 2., N, endpoint=True))) / 2

df_A = load_panda("HHG", f"{params['DIR']}/expA_laser/{params['MODEL']}", "current_density.json.gz", 
                  **hhg_params(T=params["T"], E_F=params["E_F"], v_F=params["v_F"], band_width=params["W"], 
                               field_amplitude=1., photon_energy=1., 
                               tau_diag=params["TAU_DIAG"], tau_offdiag=params["TAU_OFFDIAG"], t0=0))
df_B = load_panda("HHG", f"{params['DIR']}/expB_laser/{params['MODEL']}", "current_density.json.gz", 
                  **hhg_params(T=params["T"], E_F=params["E_F"], v_F=params["v_F"], band_width=params["W"], 
                               field_amplitude=1., photon_energy=1., 
                               tau_diag=params["TAU_DIAG"], tau_offdiag=params["TAU_OFFDIAG"], t0=0))
times = np.linspace(0, df_A["t_end"] - df_A["t_begin"], len(df_A["current_density_time"])) / (2 * np.pi)
sigma = 0.001 * params["T_AVE"] * df_A["photon_energy"] / TIME_TO_UNITLESS
__kernel = gaussian(times, times[len(times)//2], sigma)

signal_A  = -np.gradient(np.convolve(df_A["current_density_time"], __kernel, mode='same'))
signal_B  = -np.gradient(np.convolve(df_B["current_density_time"], __kernel, mode='same'))


def plot_t0(ax, t0):
    main_df = load_panda("HHG", f"{params['DIR']}/exp_laser/{params['MODEL']}", "current_density.json.gz", 
                         **hhg_params(T=params["T"], E_F=params["E_F"], v_F=params["v_F"], band_width=params["W"], 
                                      field_amplitude=1., photon_energy=1., 
                                      tau_diag=params["TAU_DIAG"], tau_offdiag=params["TAU_OFFDIAG"], t0=t0))
    signal_AB = -np.gradient(np.convolve(main_df["current_density_time"], __kernel, mode='same'))

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
    t0_unitless = t0 * df_A["photon_energy"] / TIME_TO_UNITLESS

    # --- Restrict to FFT interval and resample to uniform grids ---
    tmin, tmax = FFT_TMIN, FFT_TMAX
    sim_mask = (times >= tmin) & (times <= tmax)
    if np.sum(sim_mask) < 2:
        # fallback to entire range
        sim_mask = (times >= 0)
    # choose simulation dt from available samples in interval
    sim_dts = np.diff(times[sim_mask])
    if len(sim_dts) > 0:
        dt_sim = np.min(sim_dts)
    else:
        dt_sim = times[1] - times[0]
    uniform_t_sim = np.arange(tmin, tmax, dt_sim)
    plot_data_combined = combined_inter(uniform_t_sim, t0_unitless)
    # interpolate simulated combined and AB signals onto uniform grid
    interp_AB = interp1d(times, signal_AB, fill_value=0.0, bounds_error=False)
    signal_AB_u = interp_AB(uniform_t_sim)
    non_linear_signal = signal_AB_u - plot_data_combined

    # --- Frequency-domain plots ---
    n = len(uniform_t_sim) * 8
    fftplot = np.abs(rfft(non_linear_signal, n))**2
    fftplot /= np.max(fftplot)
    freqs = rfftfreq(n, dt_sim)
    
    ax.plot(freqs, fftplot, label=f"${t0}$ fs")

# --- Figure setup ---
fig, ax = cdf.create_frame(figsize=(8,8))
for i in range(1, MAX_FREQ, 2):
    ax.axvline(i, color="gray", ls="--")

for t0 in [0, 0.07, 0.14, 0.21, 0.28]:
    plot_t0(ax, t0)

ax.legend(loc="upper right")
ax.set_yscale("log")
ax.set_xlim(0, MAX_FREQ)
ax.set_ylim(1e-8, 2)

fig.tight_layout()

plt.show()
