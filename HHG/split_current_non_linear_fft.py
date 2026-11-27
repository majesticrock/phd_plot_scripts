import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq

from mrock.get_data import *
data_loader = DataLoader()
from mrock_centralized_scripts.legend import *

import current_density_fourier as cdf

MAX_FREQ = 14
DIR = "cascade2"
MODEL = "PiFlux"
v_F = 2.5e6
W = 600
T = 300
E_F = 118
TAU_OFFDIAG = -1
TAU_DIAG = 5
T_AVE = 40
HBAR = 0.6582119569509065

FFT_TMIN = 2 * np.pi * 2.0
FFT_TMAX = 2 * np.pi * 8.0

params = hhg_params(T=T, E_F=E_F, v_F=v_F, band_width=W,
                    field_amplitude=1., photon_energy=1.,
                    tau_diag=TAU_DIAG, tau_offdiag=TAU_OFFDIAG, t0=0)

main_df = data_loader.load_panda("HHG", f"{DIR}/exp_laser/{MODEL}", "split_current.json.gz", **params)
df_A = data_loader.load_panda("HHG", f"{DIR}/expA_laser/{MODEL}", "split_current.json.gz", **params)
df_B = data_loader.load_panda("HHG", f"{DIR}/expB_laser/{MODEL}", "split_current.json.gz", **params)

sigma = 0.001 * T_AVE * main_df["photon_energy"] / HBAR
N_times = int(main_df["N"] + 1)
times = np.linspace(0, main_df["t_end"] - main_df["t_begin"], N_times)
dt = times[1] - times[0]

def cauchy(x, mu, gamma):
    return (1. / np.pi ) * (gamma / ((x - mu)**2 + gamma**2))
kernel = cauchy(times, times[N_times // 2], sigma)

def process_signal(df, key):
    return np.convolve(df[key], kernel, mode="same")

dirac_main = process_signal(main_df, "dirac_current")
norm = np.max(np.abs(dirac_main))
dirac_main /= norm
non_dirac_main = process_signal(main_df, "non_dirac_current") / norm
dirac_A = process_signal(df_A, "dirac_current") / norm
non_dirac_A = process_signal(df_A, "non_dirac_current") / norm
dirac_B = process_signal(df_B, "dirac_current") / norm
non_dirac_B = process_signal(df_B, "non_dirac_current") / norm

inter_dirac_A = interp1d(times, dirac_A, fill_value=0.0, bounds_error=False)
inter_non_dirac_A = interp1d(times, non_dirac_A, fill_value=0.0, bounds_error=False)
inter_dirac_B = interp1d(times, dirac_B, fill_value=0.0, bounds_error=False)
inter_non_dirac_B = interp1d(times, non_dirac_B, fill_value=0.0, bounds_error=False)

def combined_inter(t, t0, inter_A, inter_B):
    if t0 >= 0:
        return inter_A(t) + inter_B(t - t0)
    return inter_A(t + t0) + inter_B(t)

sim_mask = (times >= FFT_TMIN) & (times <= FFT_TMAX)
times = times[sim_mask]
dirac_main = dirac_main[sim_mask]
non_dirac_main = non_dirac_main[sim_mask]

t0 = 0
dirac_combined_shifted = combined_inter(times, t0, inter_dirac_A, inter_dirac_B)
non_dirac_combined_shifted = combined_inter(times, t0, inter_non_dirac_A, inter_non_dirac_B)

n = len(times) * 4
freqs_scipy = rfftfreq(n, dt) * (2 * np.pi)

dirac_difference = dirac_main - dirac_combined_shifted
non_dirac_difference = non_dirac_main - non_dirac_combined_shifted
total_difference = dirac_difference + non_dirac_difference

dirac_fft = np.abs(rfft(dirac_difference, n))**2
non_dirac_fft = np.abs(rfft(non_dirac_difference, n))**2
total_fft = np.abs(rfft(total_difference, n))**2
fft_norm = np.max(total_fft)
if not np.isfinite(fft_norm) or fft_norm == 0:
    raise ValueError("Cannot normalize the FFT: the total spectrum has no finite peak.")

dirac_fft /= fft_norm
non_dirac_fft /= fft_norm
total_fft /= fft_norm

np.savetxt(
    os.path.join(os.path.dirname(__file__), "split_current_non_linear_fft.csv"),
    np.column_stack((freqs_scipy, dirac_fft, non_dirac_fft, total_fft)),
    delimiter=",",
    header="\n".join(
        [
            f"# DIR = {DIR}",
            f"# MODEL = {MODEL}",
            f"# T = {T}",
            f"# v_F (m/s) = {v_F}",
            f"# W (hbar omega_L) = {W}",
            f"# E_F (meV) = {E_F}",
            f"# tau_diag (fs) = {TAU_DIAG}",
            f"# tau_offdiag (fs) = {TAU_OFFDIAG}",
            f"# T_AVE (fs) = {T_AVE}",
            f"# FFT window (omega_L^-1) = [{FFT_TMIN}, {FFT_TMAX}]",
            "frequency (in units of omega_L),normalized Dirac contribution,normalized non-Dirac contribution,normalized total signal",
        ]
    ),
    comments="",
)

fig, ax = cdf.create_frame(figsize=(7,4.8))
ax.set_xlim(0, MAX_FREQ)
ax.set_ylim(1e-7, 2)
ax.plot(freqs_scipy, freqs_scipy**2 * dirac_fft, label="Dirac")
ax.plot(freqs_scipy, freqs_scipy**2 * non_dirac_fft, label="Non-Dirac")
ax.plot(freqs_scipy, freqs_scipy**2 * total_fft, label="Total", color="k", ls="--")

for j in range(1, MAX_FREQ, 2):
    ax.axvline(j, c="k", ls=":", alpha=0.5)

ax.legend(loc="upper right")
ax.set_xlabel(legend(r"\omega / \omega_\mathrm{L}"))
fig.tight_layout()
plt.show()
