import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import os

from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq

import current_density_time as cdt
import current_density_fourier as cdf

import mrock_centralized_scripts.path_appender as ap
ap.append()
from get_data import *
from legend import *

MAX_FREQ = 20
TIME_TO_UNITLESS = 2 * np.pi * 0.6582119569509065
FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))

# FFT window (unitless time)
FFT_TMIN = 1.0
FFT_TMAX = 9.0

# === Choose sweep parameter here ===
sweep_param = "v_F"
sweep_values = [ 2e6, 1e6, 1.5e6, 3e6]

# Default parameters in one place
PARAMS = {
    "DIR": "cascade_16",
    "MODEL": "PiFlux",
    "v_F": 1.5e6,
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

def run_and_plot(ax, ax_fft, params, color, label=None, data_norm=None):
    """Run one simulation with given params and plot results with a given color."""

    df_A = load_panda("HHG", f"{params['DIR']}/expA_laser/{params['MODEL']}", "current_density.json.gz", 
                      **hhg_params(T=params["T"], E_F=params["E_F"], v_F=params["v_F"], band_width=params["W"], 
                                   field_amplitude=1., photon_energy=1., 
                                   tau_diag=params["TAU_DIAG"], tau_offdiag=params["TAU_OFFDIAG"], t0=0))

    df_B = load_panda("HHG", f"{params['DIR']}/expB_laser/{params['MODEL']}", "current_density.json.gz", 
                      **hhg_params(T=params["T"], E_F=params["E_F"], v_F=params["v_F"], band_width=params["W"], 
                                   field_amplitude=1., photon_energy=1., 
                                   tau_diag=params["TAU_DIAG"], tau_offdiag=params["TAU_OFFDIAG"], t0=0))

    main_df = load_panda("HHG", f"{params['DIR']}/exp_laser/{params['MODEL']}", "current_density.json.gz", 
                         **hhg_params(T=params["T"], E_F=params["E_F"], v_F=params["v_F"], band_width=params["W"], 
                                      field_amplitude=1., photon_energy=1., 
                                      tau_diag=params["TAU_DIAG"], tau_offdiag=params["TAU_OFFDIAG"], t0=0))

    times = np.linspace(0, df_A["t_end"] - df_A["t_begin"], len(df_A["current_density_time"])) / (2 * np.pi)
    sigma = 0.001 * params["T_AVE"] * main_df["photon_energy"] / TIME_TO_UNITLESS
    __kernel = gaussian(times, times[len(times)//2], sigma)
    
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

    t0_unitless = 0 * main_df["photon_energy"] / TIME_TO_UNITLESS
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
    
    # --- Time-domain plots (use uniform sim time for comparisons) ---
    ax.plot(uniform_t_sim, (signal_AB_u - plot_data_combined) / np.max(np.abs(signal_AB_u - plot_data_combined)), color=color, label=f"{sweep_param}={params[sweep_param]}")

    # --- Frequency-domain plots ---
    # --- Frequency-domain plots (use resampled uniform sim grid) ---
    n = len(uniform_t_sim) * 8
    dt = dt_sim
    freqs_scipy = rfftfreq(n, dt)
    mask = freqs_scipy <= MAX_FREQ
    freqs_scipy = freqs_scipy[mask]

    fftplot = np.abs(rfft(signal_AB_u - plot_data_combined, n))**2
    label = f"{sweep_param}={params[sweep_param]}" if label is None else label
    
    if data_norm is None:
        data_norm = np.max(fftplot)
    ax_fft.plot(freqs_scipy, fftplot[mask] / data_norm, color=color, label=label)
    
    return main_df, data_norm


# --- Figure setup ---
fig, ax = cdt.create_frame()
fig_fft, ax_fft = cdf.create_frame()

ax_fft.set_yscale("log")
ax_fft.set_xlim(0, MAX_FREQ)

for i in range(1, MAX_FREQ, 2):
    ax_fft.axvline(i, color="gray", ls="--")

# Colormap setup
norm = colors.Normalize(vmin=min(sweep_values), vmax=max(sweep_values))
cmap = cm.viridis
sm = cm.ScalarMappable(norm=norm, cmap=cmap)

# --- Sweep loop ---
for i, val in enumerate(sweep_values):
    params = PARAMS.copy()
    params[sweep_param] = val  # override chosen parameter
    color = cmap(norm(val))
    if i == 0:
        main_df, data_norm = run_and_plot(ax, ax_fft, params, f"black", f"amp={val / 2e6}")#, color)
    else:
        run_and_plot(ax, ax_fft, params, f"C{i - 1}", f"amp={val / 2e6}", data_norm=data_norm)

times = np.linspace(0, main_df["t_end"] - main_df["t_begin"], len(main_df["current_density_time"])) / (2 * np.pi)
laser = np.gradient(main_df["laser_function"])

# --- Experimental data ---
EXP_PATH = "../raw_data_phd/" if os.name == "nt" else "data/"
EXPERIMENTAL_DATA = np.loadtxt(EXP_PATH + "HHG/emitted_signals_in_the_time_domain.dat").transpose()
exp_times = (15 * 0.03318960199004975 + EXPERIMENTAL_DATA[0]) * main_df["photon_energy"] / TIME_TO_UNITLESS
# Resample experimental signals onto a uniform grid inside FFT window and compute FFTs
exp_mask = (exp_times >= FFT_TMIN) & (exp_times <= FFT_TMAX)
if np.sum(exp_mask) < 2:
    exp_mask = np.ones_like(exp_times, dtype=bool)

exp_dts = np.diff(exp_times[exp_mask])
if len(exp_dts) > 0:
    dt_exp = np.min(exp_dts)
else:
    dt_exp = exp_times[1] - exp_times[0]

uniform_t_exp = np.arange(FFT_TMIN, FFT_TMAX, dt_exp)

# time-domain: plot original experimental sampling for reference
ax.plot(exp_times, -EXPERIMENTAL_DATA[4] / np.max(np.abs(EXPERIMENTAL_DATA[4])), label="Experiment", ls="--", color="k")
ax.set_xlim(FFT_TMIN, FFT_TMAX)

# interpolate experimental signal onto uniform grid for FFT
interp_exp = interp1d(exp_times, EXPERIMENTAL_DATA[4], fill_value=0.0, bounds_error=False)
exp_sig_u = interp_exp(uniform_t_exp)
n_exp = len(uniform_t_exp) * 4
exp_freqs = rfftfreq(n_exp, dt_exp)
exp_fft = np.abs(rfft(exp_sig_u, n_exp))**2
ax_fft.plot(exp_freqs, exp_fft / np.max(exp_fft), label="Experiment", ls="--", color="k")

ax.legend()
ax_fft.legend(loc="upper right")

fig.tight_layout()
fig_fft.tight_layout()

fig.subplots_adjust(hspace=0)
fig_fft.subplots_adjust(hspace=0)

plt.show()
