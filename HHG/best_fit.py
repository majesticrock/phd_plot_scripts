import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import current_density_time as cdt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

# Fixed Parameters
DIR = "cascade_new"
MODEL = "PiFlux"
v_F = 1.5e6
T = 300
E_F = 118
TAU_OFFDIAG = -1

# Parameter grids
W_values = [150, 175, 200, 225, 250, 275, 300]
TAU_DIAG_values = [10, 15, 20, 25]
T_AVE_values = [0.025, 0.035, 0.05]

import os
EXP_PATH = "../raw_data_phd_HHG/" if os.name == "nt" else "data/"
EXPERIMENTAL_DATA = np.loadtxt(f"{EXP_PATH}HHG/emitted_signals_in_the_time_domain.dat").transpose()
exp_times_raw = 14 * 0.03318960199004975 + EXPERIMENTAL_DATA[0]
exp_signals = np.array([EXPERIMENTAL_DATA[1], EXPERIMENTAL_DATA[3], EXPERIMENTAL_DATA[2]])  # A+B, A, B

NORMALIZATION_EXPERIMENT = np.max(np.abs(EXPERIMENTAL_DATA[1]))
exp_signals /= NORMALIZATION_EXPERIMENT

def gaussian(x, mu=0, sigma=1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

def compute_simulation_signal(times, df, T_AVE):
    sigma = (T_AVE * 2 * np.pi * 0.6582119569509065 / df["photon_energy"]) 
    N = int(len(times) * (T_AVE * 2 * np.pi * 0.6582119569509065 / df["photon_energy"]))
    __data = -df["current_density_time"]
    #__data = -np.diff(__data, append=0.0) / (times[1] - times[0])
    __kernel = np.ones(N)/N#gaussian(times[len(times)//4:3*len(times)//4], times[len(times)//2], sigma) #np.ones(N)/N
    
    return np.convolve(__data, __kernel, mode='same')

summed_diffs = np.zeros((len(W_values), len(TAU_DIAG_values), len(T_AVE_values)))
for i, W in enumerate(W_values):
    for j, TAU_DIAG in enumerate(TAU_DIAG_values):
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
        
        for exp_signal, df in zip(exp_signals, dfs):
            exp_times = exp_times_raw * df["photon_energy"] / (2*np.pi * 0.6582119569509065)
            times = np.linspace(0, df["t_end"] - df["t_begin"], len(df["current_density_time"])) / (2 * np.pi)
                
            for k, T_AVE in enumerate(T_AVE_values):
                signal = compute_simulation_signal(times, df, T_AVE)
                
                if k==0:
                    NORMALIZATION_SIMULATION = np.max(np.abs(signal)) 
                normalized_simulation = signal / NORMALIZATION_SIMULATION
                interpolate_simulation = np.interp(exp_times, times, normalized_simulation)

                summed_diffs[i][j][k] += np.linalg.norm(exp_signal - interpolate_simulation)

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

fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True, gridspec_kw=dict(hspace=0, wspace=0))
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

for k, (ax, exp_signal, df) in enumerate(zip(axes, exp_signals, dfs)):
    exp_times = exp_times_raw * df["photon_energy"] / (2*np.pi * 0.6582119569509065)
        
    times = np.linspace(0, df["t_end"] - df["t_begin"], len(df["current_density_time"])) / (2 * np.pi)
    signal = compute_simulation_signal(times, df, T_AVE)
    
    if k==0:
        NORMALIZATION_SIMULATION = np.max(np.abs(signal)) 
    normalized_simulation = signal / NORMALIZATION_SIMULATION
    
    ax.plot(times, normalized_simulation, label="Sim")
    ax.plot(exp_times, exp_signal, "k--", label="Exp")

axes[0].legend()
axes[-1].set_xlabel(r"$t / T_\mathrm{L}$")
axes[0].set_ylabel(r"Signal A+B")
axes[1].set_ylabel(r"Signal A")
axes[2].set_ylabel(r"Signal B")

fig.tight_layout()


from scipy.fft import rfft, rfftfreq
fig_fft, axes_fft = plt.subplots(nrows=3, sharex=True, sharey=True, gridspec_kw=dict(hspace=0, wspace=0))
OMEGA_MAX = 22

LINEAR_AB_EXP = exp_signals[1] + exp_signals[2]
PROPER_AB_EXP = exp_signals[0]
NONLINEAR_EXP = PROPER_AB_EXP - LINEAR_AB_EXP

LINEAR_AB_SIM = (compute_simulation_signal(times, dfs[1], T_AVE) + compute_simulation_signal(times, dfs[2], T_AVE)) / NORMALIZATION_SIMULATION
PROPER_AB_SIM = compute_simulation_signal(times, dfs[0], T_AVE) / NORMALIZATION_SIMULATION
NONLINEAR_SIM = PROPER_AB_SIM - LINEAR_AB_SIM

for ax, sim, exp in zip(axes_fft, [LINEAR_AB_SIM, PROPER_AB_SIM, NONLINEAR_SIM], [LINEAR_AB_EXP, PROPER_AB_EXP, NONLINEAR_EXP]):
    fft_sim = np.abs(rfft(sim))
    freqs_sim = rfftfreq(len(sim), times[1] - times[0])
    
    fft_exp = np.abs(rfft(exp))
    freqs_exp = rfftfreq(len(exp), exp_times[1] - exp_times[0])
    
    ax.plot(freqs_sim, fft_sim / np.max(fft_sim), label="Sim")
    ax.plot(freqs_exp, fft_exp / np.max(fft_exp), "k--", label="Exp")
    ax.set_yscale("log")
    ax.set_xlim(0, OMEGA_MAX)
    
    for i in range(0, OMEGA_MAX, 2):
        ax.axvline(i+1, c="k", ls=":", alpha=0.5)
    
axes_fft[0].legend()
axes_fft[-1].set_xlabel(r"$\omega / \omega_\mathrm{L}$")
axes_fft[0].set_ylabel(r"FFT linear")
axes_fft[1].set_ylabel(r"FFT proper")
axes_fft[2].set_ylabel(r"FFT nonlinear")

fig_fft.tight_layout()

plt.show()