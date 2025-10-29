import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq

import current_density_time as cdt
import current_density_fourier as cdf

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

MAX_FREQ = 20
TIME_TO_UNITLESS = 2 * np.pi * 0.6582119569509065
FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))

# === Choose sweep parameter here ===
sweep_param = "W"
sweep_values = [200]

# Default parameters in one place
PARAMS = {
    "DIR": "test",
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

    __kernel = cauchy(times, times[len(times)//2], sigma )
    #__kernel = cos_dist(int( 1e-3 * params["T_AVE"] * main_df["photon_energy"] / (times[1] - times[0])))

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
    plot_data_combined = combined_inter(times, t0_unitless)

    # --- Time-domain plots ---
    axes[0].plot(times, plot_data_combined / np.max(plot_data_combined), color=color, label=f"{sweep_param}={params[sweep_param]}")
    axes[1].plot(times, signal_AB / np.max(signal_AB), color=color, label=f"{sweep_param}={params[sweep_param]}")
    axes[2].plot(times, (signal_AB - plot_data_combined) / np.max(signal_AB - plot_data_combined), color=color, label=f"{sweep_param}={params[sweep_param]}")

    #axes[0].plot(times[:len(__kernel)], __kernel, color="red", label="Kernel")

    # --- Frequency-domain plots ---
    n = len(times) * 4
    dt = times[1] - times[0]
    freqs_scipy = rfftfreq(n, dt)

    fftplot = np.abs(rfft(plot_data_combined, n))**2
    axes_fft[0].plot(freqs_scipy, fftplot / np.max(fftplot), color=color, label=f"{sweep_param}={params[sweep_param]}")

    fftplot = np.abs(rfft(signal_AB, n))**2
    axes_fft[1].plot(freqs_scipy, fftplot / np.max(fftplot), color=color, label=f"{sweep_param}={params[sweep_param]}")

    fftplot = np.abs(rfft(signal_AB - plot_data_combined, n))**2
    axes_fft[2].plot(freqs_scipy, fftplot / np.max(fftplot), color=color, label=f"{sweep_param}={params[sweep_param]}")
    
    return main_df


# --- Figure setup ---
fig, axes = cdt.create_frame(
    nrows=3, figsize=(8.4, 8),
    ylabel_list=[
        legend(r"\partial_t j_\mathrm{lin}(t)", "a.b."),
        legend(r"\partial_t j_\mathrm{sim}(t)", "a.b."),
        legend(r"\partial_t (j_\mathrm{sim}(t) - j_\mathrm{lin}(t))", "a.b."),
    ]
)

fig_fft, axes_fft = cdf.create_frame(
    nrows=3, figsize=(8.4, 8),
    ylabel_list=[
        legend(r"|\mathrm{lin.~Signal}|^2", "a.b."),
        legend(r"|\mathrm{sim.~Signal}|^2", "a.b."),
        legend(r"|\mathrm{NL~Signal}|^2", "a.b."),
    ]
)

for ax in axes_fft:
    ax.set_yscale("log")
    ax.set_xlim(0, MAX_FREQ)

# Colormap setup
norm = colors.Normalize(vmin=min(sweep_values), vmax=max(sweep_values))
cmap = cm.viridis
sm = cm.ScalarMappable(norm=norm, cmap=cmap)

# --- Sweep loop ---
for val in sweep_values:
    params = PARAMS.copy()
    params[sweep_param] = val  # override chosen parameter
    color = cmap(norm(val))
    main_df = run_and_plot(axes, axes_fft, params, color)

times = np.linspace(0, main_df["t_end"] - main_df["t_begin"], len(main_df["current_density_time"])) / (2 * np.pi)
laser = np.gradient(main_df["laser_function"])
#axes[0].plot(times, laser / np.max(laser), c="red", ls="--")

# --- Experimental data ---
EXP_PATH = "../raw_data_phd/" if os.name == "nt" else "data/"
EXPERIMENTAL_DATA = np.loadtxt(EXP_PATH + "HHG/emitted_signals_in_the_time_domain.dat").transpose()
exp_times = (15 * 0.03318960199004975 + EXPERIMENTAL_DATA[0]) * main_df["photon_energy"] / TIME_TO_UNITLESS
exp_signals = np.array([EXPERIMENTAL_DATA[2] + EXPERIMENTAL_DATA[3], EXPERIMENTAL_DATA[1], EXPERIMENTAL_DATA[4]])

n_exp = len(exp_times)
exp_freqs = rfftfreq(n_exp, exp_times[1] - exp_times[0])

for i in range(3):
    axes[i].plot(exp_times, -exp_signals[i] / np.max(exp_signals[i]), label="Experimental data", ls="--", color="k")
    exp_fft = np.abs(rfft(exp_signals[i], n_exp))**2
    axes_fft[i].plot(exp_freqs, exp_fft / np.max(exp_fft), label="Experimental data", ls="--", color="k")

axes[0].legend()
axes_fft[0].legend(loc="upper right")

fig.tight_layout()
fig_fft.tight_layout()

fig.subplots_adjust(hspace=0)
fig_fft.subplots_adjust(hspace=0)

plt.show()
