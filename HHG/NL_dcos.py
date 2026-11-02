import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import os

from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq

import current_density_time as cdt
import current_density_fourier as cdf


from mrock.get_data import *
data_loader = DataLoader()
from mrock_centralized_scripts.legend import  *

MAX_FREQ = 12.5
TIME_TO_UNITLESS = 2 * np.pi * 0.6582119569509065
FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))

# FFT window (unitless time)
FFT_TMIN = 1.0
FFT_TMAX = 9.

# Default parameters in one place
PARAMS = [
    {
        "DIR": "cascade_2cycle",
        "MODEL": "PiFlux",
        "v_F": 1.5e6,
        "W": 450,
        "T": 300,
        "E_F": 118,
        "TAU_OFFDIAG": -1,
        "TAU_DIAG": 7,
        "T_AVE": 40,
        "field_amplitude": 1.6,
        "photon_energy": 5.25,
        "label": "2 cycles",
    },
    {
        "DIR": "cascade_8cycle",
        "MODEL": "PiFlux",
        "v_F": 1.5e6,
        "W": 450,
        "T": 300,
        "E_F": 118,
        "TAU_OFFDIAG": -1,
        "TAU_DIAG": 7,
        "T_AVE": 40,
        "field_amplitude": 1.6,
        "photon_energy": 5.25,
        "label": "8 cycles",
    }
]

T_DELAY = 0.21

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

def run_and_plot(ax_fft, params, data_norm=None):
    """Run one simulation with given params and plot results with a given"""

    df_A = data_loader.load_panda("HHG", f"{params['DIR']}/dcosA_laser/{params['MODEL']}", "current_density.json.gz", 
                      **hhg_params(T=params["T"], E_F=params["E_F"], v_F=params["v_F"], band_width=params["W"], 
                                   field_amplitude=params['field_amplitude'], photon_energy=params['photon_energy'], 
                                   tau_diag=params["TAU_DIAG"], tau_offdiag=params["TAU_OFFDIAG"], t0=0))

    df_B = data_loader.load_panda("HHG", f"{params['DIR']}/dcosB_laser/{params['MODEL']}", "current_density.json.gz", 
                      **hhg_params(T=params["T"], E_F=params["E_F"], v_F=params["v_F"], band_width=params["W"], 
                                   field_amplitude=params['field_amplitude'], photon_energy=params['photon_energy'], 
                                   tau_diag=params["TAU_DIAG"], tau_offdiag=params["TAU_OFFDIAG"], t0=0))

    main_df = data_loader.load_panda("HHG", f"{params['DIR']}/dcos_laser/{params['MODEL']}", "current_density.json.gz", 
                         **hhg_params(T=params["T"], E_F=params["E_F"], v_F=params["v_F"], band_width=params["W"], 
                                      field_amplitude=params['field_amplitude'], photon_energy=params['photon_energy'], 
                                      tau_diag=params["TAU_DIAG"], tau_offdiag=params["TAU_OFFDIAG"], t0=T_DELAY))

    times = np.linspace(0, df_A["t_end"] - df_A["t_begin"], len(df_A["current_density_time"])) / (2 * np.pi)
    sigma = 0.001 * params["T_AVE"] * main_df["photon_energy"] / TIME_TO_UNITLESS
    __kernel = cauchy(times, times[len(times)//2], sigma)
    
    signal_A  = -np.gradient(np.convolve(df_A["current_density_time"]   , __kernel, mode='same'))
    signal_B  = -np.gradient(np.convolve(df_B["current_density_time"]   , __kernel, mode='same'))
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

    t0_unitless = T_DELAY * main_df["photon_energy"] / TIME_TO_UNITLESS
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
    
    # --- Frequency-domain plots ---
    # --- Frequency-domain plots (use resampled uniform sim grid) ---
    n = len(uniform_t_sim) * 8
    dt = dt_sim
    freqs_scipy = rfftfreq(n, dt)
    mask = freqs_scipy <= MAX_FREQ
    freqs_scipy = freqs_scipy[mask]

    fftplot = np.abs(rfft(signal_AB_u - plot_data_combined, n))**2

    #if data_norm is None:
    data_norm = np.max(fftplot)
    ax_fft.plot(freqs_scipy, fftplot[mask] / data_norm, label=params["label"])
    
    return main_df, data_norm


# --- Figure setup ---
fig_fft, ax_fft = cdf.create_frame(figsize=(6.4, 8))

ax_fft.set_yscale("log")
ax_fft.set_xlim(0, MAX_FREQ)
#ax_fft.set_ylim(1e-5, 2.)

for i in range(1, int(MAX_FREQ), 2):
    ax_fft.axvline(i, color="gray", ls="--")


# --- Sweep loop ---
for i, param_set in enumerate(PARAMS):
    if i == 0:
        main_df, data_norm = run_and_plot(ax_fft, param_set)
    else:
        run_and_plot(ax_fft, param_set, data_norm=data_norm)

fig_fft.tight_layout()
fig_fft.subplots_adjust(hspace=0)
ax_fft.legend()

plt.show()
