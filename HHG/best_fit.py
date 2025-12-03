import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

# Fixed Parameters
DIR = "cascade_prec"
MODEL = "PiFlux"
v_F = 1.5e6
T = 300
E_F = 118
TAU_OFFDIAG = -1

# Parameter grids
W_values = [100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350]
TAU_DIAG_values = [10, 15, 20, 25, 30]

FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))
TIME_TO_UNITLESS = 2 * np.pi * 0.6582119569509065
T_AVE_values = 0.001 * np.array([25, 35, 50]) 

import os
EXP_PATH = "../raw_data_phd/" if os.name == "nt" else "data/"
EXPERIMENTAL_DATA = np.loadtxt(f"{EXP_PATH}HHG/emitted_signals_in_the_time_domain.dat").transpose()
exp_times_raw = 15 * 0.03318960199004975 + EXPERIMENTAL_DATA[0]
nl_exp = EXPERIMENTAL_DATA[4] / np.max(np.abs(EXPERIMENTAL_DATA[4]))

def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

def cauchy(x, mu, gamma):
    return (1. / np.pi ) * (gamma / ((x - mu)**2 + gamma**2))

def sech_distrubution(x, mu, sigma):
    return (1. / (2. * sigma)) / np.cosh(0.5 * np.pi * (x - mu) / sigma)

def laplace(x, mu, gamma):
    return 0.5 / gamma * np.exp(- np.abs(x-mu) / gamma)

def cos_dist(N):
    return (1. - np.cos(np.pi * np.linspace(0., 2., N, endpoint=True))) / 2.
    
def compute_simulation_signal(times, df, T_AVE):
    sigma = T_AVE * df["photon_energy"] / TIME_TO_UNITLESS
    N = int( T_AVE * df["photon_energy"] / (times[1] - times[0]) )
    __data = -df["current_density_time"]
    
    #__kernel = np.ones(N)/N
    #__kernel = gaussian(times, times[len(times)//2], sigma)
    #__kernel = sech_distrubution(times, times[len(times)//2], sigma)
    __kernel = cauchy(times, times[len(times)//2], sigma)
    #__kernel = laplace(times, times[len(times)//2], sigma)
    #__kernel = cos_dist(N)

    __data = np.convolve(__data, __kernel, mode='same')
    __data = -np.gradient(__data, times[1] - times[0])
    
    return __data

def compute_nonlinear(times, dfs, T_AVE):
    signal_AB = compute_simulation_signal(times, dfs[0], T_AVE)
    signal_A  = compute_simulation_signal(times, dfs[1], T_AVE)
    signal_B  = compute_simulation_signal(times, dfs[2], T_AVE)
    
    NL_signal = signal_AB - signal_A - signal_B
    NL_signal /= np.max(np.abs(NL_signal))
    return NL_signal

summed_diffs = np.zeros((len(W_values), len(TAU_DIAG_values), len(T_AVE_values)))
for i, W in enumerate(W_values):
    for j, TAU_DIAG in enumerate(TAU_DIAG_values):
        dfs = [
                load_panda("HHG", f"{DIR}/exp_laser/{MODEL}", "current_density.json.gz",
                         **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W,
                                      field_amplitude=1., photon_energy=1.,
                                      tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0), print_date=False),
                load_panda("HHG", f"{DIR}/expA_laser/{MODEL}", "current_density.json.gz",
                      **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W,
                                   field_amplitude=1., photon_energy=1.,
                                   tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0), print_date=False),
                load_panda("HHG", f"{DIR}/expB_laser/{MODEL}", "current_density.json.gz",
                      **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W,
                                   field_amplitude=1., photon_energy=1.,
                                   tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0), print_date=False)
            ]
        
        times = np.linspace(0, dfs[0]["t_end"] - dfs[0]["t_begin"], len(dfs[0]["current_density_time"])) / (2 * np.pi)
        exp_times = exp_times_raw * dfs[0]["photon_energy"] / TIME_TO_UNITLESS
        mask = (exp_times > 3) & (exp_times < 8)
        
        for k, T_AVE in enumerate(T_AVE_values):
            interpolate_simulation = np.interp(exp_times[mask], times, compute_nonlinear(times, dfs, T_AVE))
            
            summed_diffs[i][j][k] += np.linalg.norm(nl_exp[mask] - interpolate_simulation)

prev_min_index = [0, 0, 0]
prev_min_value = summed_diffs[0][0][0]
for i in range(len(W_values)):
    for j in range(len(TAU_DIAG_values)):
        for k in range(len(T_AVE_values)):
            if summed_diffs[i][j][k] < prev_min_value:
                prev_min_value = summed_diffs[i][j][k]
                prev_min_index = [i, j, k]

TAU_DIAG = TAU_DIAG_values[prev_min_index[1]]
W = W_values[prev_min_index[0]]
T_AVE = T_AVE_values[prev_min_index[2]]

print(f"Minimized difference at W={W}, tau_diag={TAU_DIAG}, t_ave={T_AVE}")
print("Difference is", prev_min_value)

fig, ax = plt.subplots()
dfs = [
        load_panda("HHG", f"{DIR}/exp_laser/{MODEL}", "current_density.json.gz",
                 **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W,
                              field_amplitude=1., photon_energy=1.,
                              tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0)),
        load_panda("HHG", f"{DIR}/expA_laser/{MODEL}", "current_density.json.gz",
              **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W,
                           field_amplitude=1., photon_energy=1.,
                           tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0)),
        load_panda("HHG", f"{DIR}/expB_laser/{MODEL}", "current_density.json.gz",
              **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W,
                           field_amplitude=1., photon_energy=1.,
                           tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0))
    ]


exp_times = exp_times_raw * dfs[0]["photon_energy"] / TIME_TO_UNITLESS
times = np.linspace(0, dfs[0]["t_end"] - dfs[0]["t_begin"], len(dfs[0]["current_density_time"])) / (2 * np.pi)
nl_simulation = compute_nonlinear(times, dfs, T_AVE)

ax.plot(times, nl_simulation, label="Sim")
ax.plot(exp_times, nl_exp, "k--", label="Exp")
#laser = np.gradient(df["laser_function"])
#ax.plot(times, laser / np.max(np.abs(laser)), "r:", label="Laser")

ax.legend()
ax.set_xlabel(r"$t / T_\mathrm{L}$")
ax.set_ylabel(r"nl. signal")

fig.tight_layout()


from scipy.fft import rfft, rfftfreq
fig_fft, ax_fft = plt.subplots()
OMEGA_MAX = 22

fft_sim = np.abs(rfft(nl_simulation))
freqs_sim = rfftfreq(len(nl_simulation), times[1] - times[0])
    
fft_exp = np.abs(rfft(nl_exp))
freqs_exp = rfftfreq(len(nl_exp), exp_times[1] - exp_times[0])
    
ax_fft.plot(freqs_sim, fft_sim / np.max(fft_sim), label="Sim")
ax_fft.plot(freqs_exp, fft_exp / np.max(fft_exp), "k--", label="Exp")
ax_fft.set_yscale("log")
ax_fft.set_xlim(0, OMEGA_MAX)

for i in range(0, OMEGA_MAX, 2):
    ax_fft.axvline(i+1, c="k", ls=":", alpha=0.5)
    
ax_fft.legend()
ax_fft.set_xlabel(r"$\omega / \omega_\mathrm{L}$")
ax_fft.set_ylabel(r"FFT")

fig_fft.tight_layout()

plt.show()