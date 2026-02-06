import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from scipy.fft import rfft, rfftfreq

import current_density_time as cdt
import current_density_fourier as cdf

import path_appender
path_appender.append()
from get_data import *
from legend import *

MAX_FREQ = 20
HBAR = 0.6582119569509065
TIME_TO_UNITLESS = 2 * np.pi * HBAR
FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))

# === Choose sweep parameter here ===
sweep_param = "W"
sweep_values = [200]

# Default parameters in one place
PARAMS = {
    "DIR": "test_new",
    "MODEL": "PiFlux",
    "v_F": 1.5e6,
    "W": 200,
    "T": 300,
    "E_F": 118,
    "TAU_OFFDIAG": -1,
    "TAU_DIAG": 10,
    "T_AVE":  50
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

def run_and_plot(axes, axes_fft, params, color):
    """Run one simulation with given params and plot results with a given color."""

    df_A = load_panda("HHG", f"{params['DIR']}/expA_laser/{params['MODEL']}", "current_density.json.gz", 
                      **hhg_params(T=params["T"], E_F=params["E_F"], v_F=params["v_F"], band_width=params["W"], 
                                   field_amplitude=1., photon_energy=1., 
                                   tau_diag=params["TAU_DIAG"], tau_offdiag=params["TAU_OFFDIAG"], t0=0))

    times = np.linspace(0, df_A["t_end"] - df_A["t_begin"], len(df_A["current_density_time"])) / (2 * np.pi)
    sigma = 0.001 * params["T_AVE"] * df_A["photon_energy"] / TIME_TO_UNITLESS

    __kernel = cauchy(times, times[len(times)//2], sigma )
    
    signal_A  = -np.gradient(np.convolve(df_A["current_density_time"]   , __kernel, mode='same'))
    axes.plot(times, signal_A / np.max(signal_A), color=color, label=f"{sweep_param}={params[sweep_param]}")

    # --- Frequency-domain plots ---
    n = len(times) * 4
    dt = times[1] - times[0]
    freqs_scipy = rfftfreq(n, dt)

    fftplot = np.abs(rfft(signal_A, n))**2
    axes_fft.plot(freqs_scipy, fftplot / np.max(fftplot), color=color, label=f"{sweep_param}={params[sweep_param]}")
    
    return df_A


# --- Figure setup ---
fig, ax = cdt.create_frame()
fig_fft, ax_fft = cdf.create_frame()

ax_fft.set_yscale("log")
ax_fft.set_xlim(0, MAX_FREQ)

# Colormap setup
norm = colors.Normalize(vmin=min(sweep_values), vmax=max(sweep_values))
cmap = cm.viridis
sm = cm.ScalarMappable(norm=norm, cmap=cmap)

# --- Sweep loop ---
for val in sweep_values:
    params = PARAMS.copy()
    params[sweep_param] = val  # override chosen parameter
    color = cmap(norm(val))
    main_df = run_and_plot(ax, ax_fft, params, color)

times = np.linspace(0, main_df["t_end"] - main_df["t_begin"], len(main_df["current_density_time"])) / (2 * np.pi)
laser = (np.gradient(main_df["laser_function"]))
ax.plot(times, laser / np.max(np.abs(laser)), c="red", ls="--")

# --- Experimental data ---
EXP_PATH = "../raw_data_phd/" if os.name == "nt" else "data/"
#LASER_DATA = np.loadtxt(EXP_PATH + "HHG/pulse_AB.dat").transpose()
#laser_times = (7 * 0.03318960199004975 + LASER_DATA[0]) * main_df["photon_energy"] / TIME_TO_UNITLESS
#ax.plot(laser_times, -LASER_DATA[1] / np.max(np.abs(LASER_DATA[1])), c="k", ls=":")

EXPERIMENTAL_DATA = np.loadtxt(EXP_PATH + "HHG/emitted_signals_in_the_time_domain.dat").transpose()
exp_times = (7 * 0.03318960199004975 + EXPERIMENTAL_DATA[0]) * main_df["photon_energy"] / TIME_TO_UNITLESS
exp_signal = EXPERIMENTAL_DATA[3]

n_exp = len(exp_times)
exp_freqs = rfftfreq(n_exp, exp_times[1] - exp_times[0])

ax.plot(exp_times, -exp_signal / np.max(exp_signal), label="Experimental data", ls="--", color="k")
exp_fft = np.abs(rfft(exp_signal, n_exp))**2
ax_fft.plot(exp_freqs, exp_fft / np.max(exp_fft), label="Experimental data", ls="--", color="k")

ax.legend()
ax_fft.legend(loc="upper right")

fig.tight_layout()
fig_fft.tight_layout()

fig.subplots_adjust(hspace=0)
fig_fft.subplots_adjust(hspace=0)

plt.show()
