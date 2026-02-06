import numpy as np
import matplotlib.pyplot as plt

import mrock_centralized_scripts.path_appender as ap
ap.append()
from get_data import *
from legend import *

FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))
HBAR = 0.6582119569509065

# Default parameters in one place
params = {
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


def make_times(df):
    N = len(df["current_density_time"])
    return np.linspace(0, df["t_end"] - df["t_begin"], N) * HBAR / df["photon_energy"]
def make_kernel(times, T_AVE, kernel_fn=cauchy):
    sigma = 0.001 * T_AVE
    return kernel_fn(times, times[len(times)//2], sigma)


def process_current_df(df, T_AVE, method="ddtj"):
    times = make_times(df)
    kernel = make_kernel(times, T_AVE)

    if method == "ddtj":
        ddtj = -np.convolve(np.asarray(df["current_density_time"]), kernel, mode="same")
        jt = np.cumsum(ddtj)
    elif method == "jt":
        jt = -np.convolve(np.asarray(df["current_density_time"]), kernel, mode="same")
        ddtj = np.gradient(jt)
    else:
        raise ValueError("unknown method")

    return times, ddtj / np.max(np.abs(ddtj))


def compute_normalized_fft(times, signal, n=None):
    """
    Compute normalized FFT magnitude of `signal` sampled at `times`.
    Returns (freqs, fft_mag) where fft_mag is normalized to max=1.
    """
    times = np.asarray(times)
    signal = np.asarray(signal)
    if n is None:
        n = len(times)
    if len(times) < 2:
        dt = 1.0
    else:
        dt = times[1] - times[0]
    freqs = np.fft.rfftfreq(n, dt)
    fft_mag = np.abs(np.fft.rfft(signal, n=n))
    maxv = fft_mag.max() if fft_mag.size else 1.0
    if maxv == 0:
        norm_fft = fft_mag
    else:
        norm_fft = fft_mag / maxv
    return freqs, norm_fft


fig, ax = plt.subplots()

# list of (data_dir_prefix, label, method)
variants = [
    #("test_ddtj0/expA_laser", "ddt-j-0 $\\partial_t j(t)$", "ddtj"),
    ("test_ddtj1/expA_laser", "ddt-j-1 $\\partial_t j(t)$", "ddtj"),
    ("test_ddtj2/expA_laser", "ddt-j-2 $\\partial_t j(t)$", "ddtj"),
    ("test_ddtj16/expA_laser", "ddt-j-16 $\\partial_t j(t)$", "ddtj"),
    #("test_ddtjk/expA_laser", "ddt-j-k $\\partial_t j(t)$", "ddtj"),
    #("test_basek/expA_laser", "base-k $\\partial_t j(t)$", "jt"),
    ("test_base/expA_laser", "base $\\partial_t j(t)$", "jt"),
]


fft_results = []

for data_prefix, label, method in variants:
    main_df = load_panda(
        "HHG",
        f"{data_prefix}/{params['MODEL']}",
        "current_density.json.gz",
        **hhg_params(
            T=params["T"],
            E_F=params["E_F"],
            v_F=params["v_F"],
            band_width=params["W"],
            field_amplitude=1.,
            photon_energy=1.,
            tau_diag=params["TAU_DIAG"],
            tau_offdiag=params["TAU_OFFDIAG"],
            t0=0
        )
    )
    times, ddtj_norm = process_current_df(main_df, params["T_AVE"], method=method)
    ax.plot(times, ddtj_norm, label=label)
    
    freqs, fft_norm = compute_normalized_fft(times, ddtj_norm)
    fft_results.append((freqs, fft_norm, label))

fig_fft, ax_fft = plt.subplots()
for freqs, fft_norm, label in fft_results:
    ax_fft.plot(freqs, fft_norm, label=label)
    
# --- Experimental data ---
EXP_PATH = "../raw_data_phd/" if os.name == "nt" else "data/"
EXPERIMENTAL_DATA = np.loadtxt(EXP_PATH + "HHG/emitted_signals_in_the_time_domain.dat").transpose()
exp_times = 7 * 0.03318960199004975 + EXPERIMENTAL_DATA[0] # 0.03318960199004975 is the exp dt; we added 7 "zeros" before the laser actually starts
exp_signal = EXPERIMENTAL_DATA[3]
n_exp = len(exp_times)
ax.plot(exp_times, -exp_signal / np.max(np.abs(exp_signal)), "k--", label="Emitted $E(t)$")
ax.legend()    

# --- plot FFTs of experimental data ---
freqs_exp, fft_exp = compute_normalized_fft(exp_times, exp_signal)

ax_fft.plot(freqs_exp, fft_exp, "k--", label="Exp FFT")


ax_fft.set_xlabel("$\\omega$ (THz)")
ax_fft.set_ylabel("Normalized FFT")
ax_fft.set_xlim(0, 20)
ax_fft.set_yscale('log')
ax_fft.legend()
fig_fft.tight_layout()


ax.set_ylabel("Signal")
ax.set_xlabel("$t$ (ps)")
ax.legend()
fig.tight_layout()

plt.show()