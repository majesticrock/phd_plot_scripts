import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import current_density_time as cdt

import mrock_centralized_scripts.path_appender as ap
ap.append()
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
W_values = [225, 250, 275, 300]
TAU_DIAG_values = [10, 15, 20]
T_AVE_values = [0.025, 0.035, 0.05]

import os
EXP_PATH = "../raw_data_phd/" if os.name == "nt" else "data/"
EXPERIMENTAL_DATA = np.loadtxt(f"{EXP_PATH}HHG/emitted_signals_in_the_time_domain.dat").transpose()
exp_times_raw = 16 * 0.03318960199004975 + EXPERIMENTAL_DATA[0]
exp_signals = np.array([EXPERIMENTAL_DATA[3], EXPERIMENTAL_DATA[2], EXPERIMENTAL_DATA[1]])  # A, B, A+B

def gaussian(x, mu=0, sigma=1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

def compute_simulation_signal(times, df, T_AVE):
    sigma = (T_AVE * 2 * np.pi * 0.6582119569509065 / df["photon_energy"]) 
    N = int(len(times) * (T_AVE * 2 * np.pi * 0.6582119569509065 / df["photon_energy"]))
    __data = -df["current_density_time"]
    #__data = -np.diff(__data, append=0.0) / (times[1] - times[0])
    __kernel = gaussian(times[len(times)//4:3*len(times)//4], times[len(times)//2], sigma) #np.ones(N)/N
    
    return np.convolve(__data, __kernel, mode='same')

def create_figure_and_plot(signal_index, title_label):
    fig, axes = plt.subplots(len(W_values), len(TAU_DIAG_values), figsize=(16, 12), sharex=True, sharey=True)
    axes = np.array(axes).reshape(len(W_values), len(TAU_DIAG_values))

    for i, W in enumerate(W_values):
        for j, TAU_DIAG in enumerate(TAU_DIAG_values):
            ax = axes[i][j]
            if signal_index == 0:
                df = load_panda("HHG", f"{DIR}/expA_laser/{MODEL}", "current_density.json.gz",
                              **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W,
                                           field_amplitude=1., photon_energy=1.,
                                           tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0))
            elif signal_index == 1:
                df = load_panda("HHG", f"{DIR}/expB_laser/{MODEL}", "current_density.json.gz",
                              **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W,
                                           field_amplitude=1., photon_energy=1.,
                                           tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0))
            else:
                df = load_panda("HHG", f"{DIR}/exp_laser/{MODEL}", "current_density.json.gz",
                                 **hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W,
                                              field_amplitude=1., photon_energy=1.,
                                              tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0))
                
            for T_AVE in T_AVE_values:
                times = np.linspace(0, df["t_end"] - df["t_begin"], len(df["current_density_time"])) / (2 * np.pi)
                signal = compute_simulation_signal(times, df, T_AVE)
     
                label = f"$t_\\mathrm{{ave}}={T_AVE}$"
                ax.plot(times, signal / np.max(signal), label=label)

            # Plot experiment
            exp_times = exp_times_raw * df["photon_energy"] / (2*np.pi * 0.6582119569509065)
            exp_signal = exp_signals[signal_index]
            ax.plot(exp_times, exp_signal / np.max(exp_signal), linestyle='--', color='black', label="Experiment")

            # Title and labels
            ax.set_title(f"$W={W}$, $\\tau_\\mathrm{{diag}}={TAU_DIAG}$", fontsize=10)
            if j == 0:
                ax.set_ylabel("j(t) [norm.]")
            if i == len(W_values) - 1:
                ax.set_xlabel("t [cycles]")
            ax.legend(fontsize='x-small')

    fig.suptitle(f"{title_label} comparison across $W$, $\\tau_\\mathrm{{diag}}$, $t_\\mathrm{{ave}}$", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

# Create and display three figures
create_figure_and_plot(0, r"$j_A(t)$")
create_figure_and_plot(1, r"$j_B(t)$")
create_figure_and_plot(2, r"$j_{A+B}(t)$")
plt.show()