import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq

import current_density_time as cdt
import current_density_fourier as cdf


from mrock.get_data import *
data_loader = DataLoader()
from mrock_centralized_scripts.legend import  *

MAX_FREQ = 14
HBAR = 0.6582119569509065
FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))

# FFT window (unitless time)
FFT_TMIN = 2*np.pi * 3.0
FFT_TMAX = 2*np.pi * 8.0

# Default parameters in one place
params = {
    "DIR": "cascade2",
    "MODEL": "PiFlux",
    "v_F": 2.5e6,
    "W": 600,
    "T": 300,
    "E_F": 118,
    "TAU_OFFDIAG": -1,
    "TAU_DIAG": 5,
    "T_AVE": 40,
    "field_amplitude": 1
}

T_DELAY = 0.0

def gaussian(x, mu, FWHM):
    gamma = FWHM / FWHM_TO_SIGMA
    return (1 / (gamma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * gamma**2))
def cauchy(x, mu, gamma):
    return (1. / np.pi ) * (gamma / ((x - mu)**2 + gamma**2))
def laplace(x, mu, gamma):
    return np.log(2.) / gamma * np.exp(- 2*np.log(2) / gamma * np.abs(x-mu))
def sech_distrubution(x, mu, gamma):
    return (1. / (2. * gamma)) / np.cosh(0.5 * np.pi * (x - mu) / gamma)
def cos_dist(N):
    return (1. - np.cos(np.pi * np.linspace(0., 2., N, endpoint=True))) / 2

df_A = data_loader.load_panda("HHG", f"{params['DIR']}/expA_laser/{params['MODEL']}", "current_density.json.gz", 
                  **hhg_params(T=params["T"], E_F=params["E_F"], v_F=params["v_F"], band_width=params["W"], 
                               field_amplitude=1., photon_energy=1., 
                               tau_diag=params["TAU_DIAG"], tau_offdiag=params["TAU_OFFDIAG"], t0=0))
df_B = data_loader.load_panda("HHG", f"{params['DIR']}/expB_laser/{params['MODEL']}", "current_density.json.gz", 
                  **hhg_params(T=params["T"], E_F=params["E_F"], v_F=params["v_F"], band_width=params["W"], 
                               field_amplitude=1., photon_energy=1., 
                               tau_diag=params["TAU_DIAG"], tau_offdiag=params["TAU_OFFDIAG"], t0=0))
main_df = data_loader.load_panda("HHG", f"{params['DIR']}/exp_laser/{params['MODEL']}", "current_density.json.gz", 
                     **hhg_params(T=params["T"], E_F=params["E_F"], v_F=params["v_F"], band_width=params["W"], 
                                  field_amplitude=1., photon_energy=1., 
                                  tau_diag=params["TAU_DIAG"], tau_offdiag=params["TAU_OFFDIAG"], t0=T_DELAY))

# time in the simulation is measured in 1/(hbar omega_L) and natural units (hbar=1)
times = np.linspace(0, df_A["t_end"] - df_A["t_begin"], len(df_A["current_density_time"]))
sigma = 0.001 * params["T_AVE"] * main_df["photon_energy"] / HBAR
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
t0_unitless = T_DELAY * main_df["photon_energy"] / HBAR
# --- Restrict to FFT interval and resample to uniform grids ---
tmin, tmax = FFT_TMIN, FFT_TMAX
sim_mask = (times >= tmin) & (times <= tmax)
if np.sum(sim_mask) < 2:
    # fallback to entire range
    sim_mask = (times >= 0)
# choose simulation dt from available samples in interval
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
freqs = rfftfreq(n, dt_sim) * (2*np.pi)
np.savetxt(
    os.path.join(os.path.dirname(__file__), "simulation_fft_data.csv"),
    np.column_stack((freqs, fftplot)),
    delimiter=",",
    header="\n".join(
        [
            f"# DIR = {params['DIR']}",
            f"# MODEL = {params['MODEL']}",
            f"# T = {params['T']}",
            f"# v_F (m/s) = {params['v_F']}",
            f"# W (hbar omega_L) = {params['W']}",
            f"# E_F (meV) = {params['E_F']}",
            f"# tau (fs) = {params['TAU_DIAG']}",
            f"# T_AVE (fs) = {params['T_AVE']}",
            f"# T_DELAY = {T_DELAY} fs",
            "frequency (in units of omega_L),normalized non-linear signal",
        ]
    ),
    comments="",
)

# --- Figure setup ---
fig, ax = cdf.create_frame(figsize=(7,4.8))
ax.plot(freqs, fftplot, label="Simulation")
ax.set_ylim(1e-6, 2)
# --- Experimental data ---
if T_DELAY == 0:
    EXP_PATH = "../raw_data_phd/" if os.name == "nt" else "data/"
    EXPERIMENTAL_DATA = np.loadtxt(EXP_PATH + "HHG/emitted_signals_in_the_time_domain.dat").transpose()
    exp_times = (7.5 * 0.03318960199004975 + EXPERIMENTAL_DATA[0]) * main_df["photon_energy"] / HBAR

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

    # interpolate experimental signal onto uniform grid for FFT
    interp_exp = interp1d(exp_times, EXPERIMENTAL_DATA[4], fill_value=0.0, bounds_error=False)
    exp_sig_u = interp_exp(uniform_t_exp)
    n_exp = len(uniform_t_exp) * 4
    exp_freqs = rfftfreq(n_exp, dt_exp) * (2*np.pi)
    exp_fft = np.abs(rfft(exp_sig_u, n_exp))**2
    exp_fft /= np.max(exp_fft)
    ax.plot(exp_freqs, exp_fft , label="Experiment", ls="--", color="k")


if T_DELAY == 0:
    peaks = [1, 2.7, 4.4, 5.7, 7, 8.3, 9.4, 11.5, 13.2]
else:
    peaks = [1, 2.82, 4.73, 7.06, 8.7, 9.5, 11.3]
    
for i, peak in enumerate(peaks, start=1):
    ax.axvline(peak, ls="--", c="gray")
    ax.text(
        peak,
        0.3 if peak < 8. else 0.75,
        f"$({i}): {peak} \\omega_\\mathrm{{L}}$",
        rotation=90,
        va="top",
        ha="right",
        transform=ax.get_xaxis_transform()
    )

ax.legend(loc="upper right")
ax.set_yscale("log")
ax.set_xlim(0, MAX_FREQ)

fig.tight_layout()


fig_time, ax_time = cdt.create_frame(ylabel_list=r"$\partial_t j_\mathrm{NL}(t)$")
ax_time.plot(uniform_t_sim, non_linear_signal / np.max(np.abs(non_linear_signal)))
if T_DELAY==0:
    ax_time.plot(exp_times[exp_mask], EXPERIMENTAL_DATA[4][exp_mask] / np.max(np.abs(EXPERIMENTAL_DATA[4])), ls="--", c="k")

plt.show()
