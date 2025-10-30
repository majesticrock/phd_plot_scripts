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

fig, ax = plt.subplots()

main_df = load_panda("HHG", f"{params['DIR']}/expA_laser/{params['MODEL']}", "current_density.json.gz", 
                  **hhg_params(T=params["T"], E_F=params["E_F"], v_F=params["v_F"], band_width=params["W"], 
                               field_amplitude=1., photon_energy=1., 
                               tau_diag=params["TAU_DIAG"], tau_offdiag=params["TAU_OFFDIAG"], t0=0))

N=len(main_df["current_density_time"])

times = np.linspace(0, main_df["t_end"] - main_df["t_begin"],N) * HBAR / main_df["photon_energy"]
sigma = 0.001 * params["T_AVE"]
__kernel = cauchy(times, times[N//2], sigma )

j_t = -(main_df["current_density_time"])
signal  = np.cumsum(j_t)
#ax.plot(times, j_t / np.max(np.abs(j_t)), label=f"Simulation $j(t)$")
#ax.plot(times, signal / np.max(np.abs(signal)), label=f"Simulation $\\partial_t j(t)$")

ax.plot(times, j_t / np.max(np.abs(j_t)), label=f"Simulation $\\partial_t j(t)$")
ax.plot(times, signal / np.max(np.abs(signal)), label=f"Simulation $j(t)$")

laser = -np.gradient(main_df["laser_function"])
ax.plot(times, laser / np.max(np.abs(laser)), label="Laser $E(t)$", c="k", ls="--")

ax.set_ylabel("Signal")
ax.set_xlabel("$t$ (ps)")
ax.legend()
fig.tight_layout()

plt.show()
