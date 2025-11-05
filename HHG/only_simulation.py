import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

from scipy.fft import rfft, rfftfreq

FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))
HBAR = 0.6582119569509065

params = {
    "DIR": "test",
    "MODEL": "PiFlux",
    "v_F": 1e5,
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

def plot_time_and_fft(df, ax, ax_fft, sigma, label="", plot_j=True, color=None):
    N = len(df["current_density_time"])
    time_axis = np.linspace(0, df["t_end"] - df["t_begin"], N) * HBAR / df["photon_energy"]
    dt = time_axis[1] - time_axis[0]
    
    # Compute signals
    __kernel = cauchy(time_axis, time_axis[N//2], sigma)
    j_t = -np.convolve((df["current_density_time"]), __kernel, mode="same")
    ddtj = np.gradient(j_t, dt)
    
    # Plot time domain
    ax.plot(time_axis, ddtj / np.max(np.abs(ddtj)), label=f"{label}$\\partial_t j(t)$", color=color)
    
    if plot_j:
        ax.plot(time_axis, j_t / np.max(np.abs(j_t)), label=f"{label}$j(t)$", color=color)
        laser = -np.gradient(df["laser_function"])
        ax.plot(time_axis, laser / np.max(np.abs(laser)), label=f"{label}Laser $E(t)$", c="k", ls="--")
    
    # FFT analysis
    n = len(time_axis)
    fft_axis = rfftfreq(n, d=dt) * HBAR
    
    fft_ddtj = np.abs(rfft(ddtj, n=n))**2
    fft_ddtj = fft_ddtj / fft_ddtj.max()
    ax_fft.plot(fft_axis, fft_ddtj, label=f"{label}", color=color)
    
    return (j_t, ddtj, fft_ddtj)

def format_plot(fig, ax, fig_fft, ax_fft):
    ax.set_ylabel("Signal")
    ax.set_xlabel("$t$ (ps)")
    ax.legend()
    
    ax_fft.set_xlim(0, 30)
    ax_fft.set_yscale("log")
    for i in range(1, int(ax_fft.get_xlim()[1]), 2):
        ax_fft.axvline(i, c="k", alpha=0.6, ls="--", zorder=-5)
    
    ax_fft.set_xlabel(r"$\omega / \omega_\mathrm{L}$")
    ax_fft.set_ylabel("Normalized FFT")
    ax_fft.legend()
    
    fig.tight_layout()
    fig_fft.tight_layout()

# Example usage:
if __name__ == "__main__":
    fig, ax = plt.subplots()
    fig_fft, ax_fft = plt.subplots()
    
    main_df = load_panda("HHG", f"{params['DIR']}/dgaussA_laser/{params['MODEL']}", 
                        "current_density.json.gz", 
                        **hhg_params(T=params["T"], E_F=params["E_F"], 
                                   v_F=1.5e6, band_width=params["W"], 
                                   field_amplitude=1., photon_energy=1., 
                                   tau_diag=params["TAU_DIAG"], 
                                   tau_offdiag=params["TAU_OFFDIAG"], t0=0))
    plot_time_and_fft(main_df, ax, ax_fft, 0.001 * params["T_AVE"], label="$W = 200 \\hbar \\omega_L$", plot_j=False)
    
    main_df = load_panda("HHG", f"{params['DIR']}/dgaussA_laser/{params['MODEL']}", 
                        "current_density.json.gz", 
                        **hhg_params(T=params["T"], E_F=params["E_F"], 
                                   v_F=1.5e6, band_width=1000, 
                                   field_amplitude=1., photon_energy=1., 
                                   tau_diag=params["TAU_DIAG"], 
                                   tau_offdiag=params["TAU_OFFDIAG"], t0=0))
    plot_time_and_fft(main_df, ax, ax_fft, 0.001 * params["T_AVE"], label="$W = 1000 \\hbar \\omega_L$", plot_j=False)
    
    main_df = load_panda("HHG", f"{params['DIR']}/dgaussA_laser/{params['MODEL']}", 
                        "current_density.json.gz", 
                        **hhg_params(T=params["T"], E_F=params["E_F"], 
                                   v_F=1.5e6, band_width=2000, 
                                   field_amplitude=1., photon_energy=1., 
                                   tau_diag=params["TAU_DIAG"], 
                                   tau_offdiag=params["TAU_OFFDIAG"], t0=0))
    plot_time_and_fft(main_df, ax, ax_fft, 0.001 * params["T_AVE"], label="$W = 2000 \\hbar \\omega_L$", plot_j=False)
    
    
    
    format_plot(fig, ax, fig_fft, ax_fft)
    plt.show()