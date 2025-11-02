import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))
HBAR = 0.6582119569509065

# Default parameters in one place
params = {
    "DIR": "test",
    "MODEL": "PiFlux",
    "v_F": 1.5e3,
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

fig, ax = plt.subplots()

main_df = load_panda("HHG", f"{params['DIR']}/dgaussA_laser/{params['MODEL']}", "current_density.json.gz", 
                  **hhg_params(T=params["T"], E_F=params["E_F"], v_F=params["v_F"], band_width=params["W"], 
                               field_amplitude=1., photon_energy=1., 
                               tau_diag=params["TAU_DIAG"], tau_offdiag=params["TAU_OFFDIAG"], t0=0))

N=len(main_df["current_density_time"])

times = np.linspace(0, main_df["t_end"] - main_df["t_begin"],N) * HBAR / main_df["photon_energy"]
sigma = 0.001 * params["T_AVE"]
__kernel = cauchy(times, times[N//2], sigma )

j_t = -np.convolve((main_df["current_density_time"]), __kernel, mode="same")
ddtj  = np.gradient(j_t)

ax.plot(times, j_t / np.max(np.abs(j_t)), label=f"Simulation $j(t)$")
ax.plot(times, ddtj / np.max(np.abs(ddtj)), label=f"Simulation $\\partial_t j(t)$")

laser = -np.gradient(main_df["laser_function"])
ax.plot(times, laser / np.max(np.abs(laser)), label="Laser $E(t)$", c="k", ls="--")

ax.set_ylabel("Signal")
ax.set_xlabel("$t$ (ps)")
ax.legend()
fig.tight_layout()


# FFT of ddtj
dt = times[1] - times[0]
n = len(times)
freqs = (np.fft.rfftfreq(n, d=dt) * HBAR) 
fft_ddtj = np.abs(np.fft.rfft(ddtj, n=n))**2
fft_ddtj = fft_ddtj / fft_ddtj.max()

fig_fft, ax_fft = plt.subplots()
ax_fft.plot(freqs, fft_ddtj, label=r'$\partial_t j$')
ax_fft.set_xlim(0, freqs.max())
ax_fft.set_xlabel(r"$\omega / \omega_\mathrm{L}$")
ax_fft.set_ylabel("Normalized FFT")
ax_fft.legend()
ax_fft.set_yscale("log")

for i in range(1, int(ax_fft.get_xlim()[1]), 2):
    ax_fft.axvline(i, c="k", alpha=0.6, ls="--", zorder=-5)

fig_fft.tight_layout()

plt.show()
