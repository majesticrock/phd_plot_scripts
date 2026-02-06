import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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

DIR = "cascade_prec"
MODEL = "PiFlux"
v_F = 1.5e6
W = 200
TAU_DIAG = 15
MAX_FREQ = 10

TIME_TO_UNITLESS = 2 * np.pi * 0.6582119569509065
T_AVE = 50e-3

df_A = load_panda("HHG", f"{DIR}/expA_laser/{MODEL}", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=v_F, band_width=W, field_amplitude=1., photon_energy=1., tau_diag=TAU_DIAG, tau_offdiag=-1, t0=0))
df_B = load_panda("HHG", f"{DIR}/expB_laser/{MODEL}", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=v_F, band_width=W, field_amplitude=1., photon_energy=1., tau_diag=TAU_DIAG, tau_offdiag=-1, t0=0))

times = np.linspace(0, df_A["t_end"] - df_A["t_begin"], len(df_A["current_density_time"])) / (2 * np.pi)

def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))
def cauchy(x, mu, gamma):
    return (1. / np.pi ) * (gamma / ((x - mu)**2 + gamma**2))
def sech_distrubution(x, mu, sigma):
    return (1. / (2. * sigma)) / np.cosh(0.5 * np.pi * (x - mu) / sigma)
def laplace(x, mu, gamma):
    return 0.5 / gamma * np.exp(- np.abs(x-mu) / gamma)

def compute_simulation_signal(t, signal):
    sigma = T_AVE * df_A["photon_energy"] / TIME_TO_UNITLESS
    N = int( T_AVE / (t[1] - t[0]) )

    #__kernel = np.ones(N)/N
    #__kernel = gaussian(times, times[len(times)//2], sigma)
    #__kernel = sech_distrubution(times, times[len(times)//2], sigma)
    __kernel = cauchy(t, t[len(t)//2], sigma)
    #__kernel = laplace(times, times[len(times)//2], sigma)

    __data = np.convolve(signal, __kernel, mode='same')
    __data = -np.gradient(__data, t[1] - t[0])
    return __data


signal_A = compute_simulation_signal(times, df_A["current_density_time"])
signal_B = compute_simulation_signal(times, df_B["current_density_time"])

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

fig, axes = cdt.create_frame(nrows=3, figsize=(6.4, 8), 
                             ylabel_list=[legend(r"j_\mathrm{lin}(t)", "arb. units"), 
                                          legend(r"j_\mathrm{sim}(t)", "arb. units"), 
                                          legend(r"j_\mathrm{sim}(t) - j_\mathrm{lin}(t)", "arb. units")])#fig_laser, ax_laser = plt.subplots()

fig_fft, axes_fft = plt.subplots(ncols=3, figsize=(12.8, 8), sharex=True, sharey=True)
for ax in axes_fft:
    ax.set_yscale('log')
    ax.set_xlim(0, MAX_FREQ)

axes_fft[0].set_ylabel(legend(r"|\omega j_\mathrm{lin}(\omega)|^2", "arb. units"))
axes_fft[1].set_ylabel(legend(r"|\omega j_\mathrm{sim}(\omega)|^2", "arb. units"))
axes_fft[2].set_ylabel(legend(r"\omega^2 |j_\mathrm{sim}(\omega) - j_\mathrm{lin}(\omega)|^2", "arb. units"))
#fig_fft.tight_layout()

fig.subplots_adjust(hspace=0)
fig_fft.subplots_adjust(hspace=0)

t0_values = [0, 0.07, 0.14, 0.21, 0.28, 0.35, 0.42, 0.49, 0.56]#, 0.63, 0.7
norm = Normalize(vmin=min(t0_values), vmax=max(t0_values))
cmap = mpl.colormaps["viridis"]
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

for i, t0 in enumerate(t0_values):
    color = cmap(norm(t0))
    main_df = load_panda("HHG", f"{DIR}/exp_laser/{MODEL}", "current_density.json.gz", 
                        **hhg_params(T=300, E_F=118, v_F=v_F, band_width=W, field_amplitude=1., photon_energy=1., tau_diag=TAU_DIAG, tau_offdiag=-1, t0=t0))
    times2 = np.linspace(0, main_df["t_end"] - main_df["t_begin"], len(main_df["current_density_time"])) / (2 * np.pi)
    
    t0_unitless = t0 * main_df["photon_energy"] / TIME_TO_UNITLESS
    signal_linear = combined_inter(times2, t0_unitless)
    signal_simulation = compute_simulation_signal(times2, main_df["current_density_time"])
    signal_non_linear = signal_simulation - signal_linear
    
    axes[0].plot(times2, signal_linear, color=color, label=f"$t_0 = {t0}$ ps")
    axes[1].plot(times2, signal_simulation, color=color, label=f"$t_0 = {t0}$ ps")
    axes[2].plot(times2, signal_non_linear, color=color, label=f"$t_0 = {t0}$ ps")
    
    n = len(times2) * 4
    dt = times2[1] - times2[0]
    freqs_scipy = rfftfreq(n, dt)
    
    fftplot = np.abs(rfft(signal_linear, n))**2
    axes_fft[0].plot(freqs_scipy, 10**(2*i) * fftplot , label=f"$t_0 = {t0}$ ps", color=color)
    
    fftplot = np.abs(rfft(signal_simulation, n))**2
    axes_fft[1].plot(freqs_scipy, 10**(2*i) * fftplot, color=color)
    
    fftplot = np.abs(rfft(signal_non_linear, n))**2
    axes_fft[2].plot(freqs_scipy, 10**(2*i) * fftplot, color=color)

    cdf.add_verticals(freqs_scipy, axes_fft[0], max_freq=MAX_FREQ)
    cdf.add_verticals(freqs_scipy, axes_fft[1], max_freq=MAX_FREQ)
    cdf.add_verticals(freqs_scipy, axes_fft[2], max_freq=MAX_FREQ)

    #ax_laser.plot(times2, combined_laser(times2, t0_unitless), c=f"C{i}")
    #ax_laser.plot(times2, main_df["laser_function"], ls="--", c=f"C{i}", linewidth=4)

cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04)
cbar.set_label(r"$t_0$ (ps)")

fft_cbar = fig_fft.colorbar(sm, ax=axes_fft, fraction=0.046, pad=0.04)
fft_cbar.set_label(r"$t_0$ (ps)")

for ax in axes_fft:
    ax.set_ylim(1, 1e25)
    ax.set_xlabel(legend(r"\omega / \omega_\mathrm{L}"))

plt.show()