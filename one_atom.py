import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

A0 = 12.
OMEGA_L = 1.
N_LASER = 4
OMEGA_ENV = OMEGA_L / N_LASER
T_MAX = 2 * N_LASER * np.pi / OMEGA_L
N_K = 200
N_T = 2000
RELAX = 1.
MAX_FREQ = 20
BETA = 5

ks = np.linspace(-np.pi, np.pi, N_K, endpoint=False)
t_eval = np.linspace(0.0, T_MAX, N_T)

def vector_potential(t):
    return 0.5 * A0 * np.cos(OMEGA_L * t) * (1. - np.cos(OMEGA_ENV * t))
def fermi_function(epsilon):
    return 1. / (np.exp(BETA * epsilon) + 1.)

if __name__ == "__main__":
    def right_side(t, y, k):
        k_eff = k - vector_potential(t)
        equilibrium = fermi_function(-2. * np.cos(k_eff))
        return RELAX * (equilibrium - y)

    current = np.zeros_like(t_eval)
    for k in ks:
        solution = solve_ivp(right_side, [0, T_MAX], [fermi_function(-2. * np.cos(k))], 
                             method="DOP853", t_eval=t_eval, args=(k,))

        current += solution.y[:,0] * np.sin(k - vector_potential(t_eval))
    current *= 1. / N_K # J=1

    fig_j, ax_j = plt.subplots()
    ax_j.plot(t_eval * OMEGA_L, current, label="$j_1(t)$")
    ax_j.set_xlabel(r'$t / T_L$')
    ax_j.set_ylabel(r'Signal')
    ax_j.legend()
    fig_j.tight_layout()

    ###################################################

    dt = t_eval[1] - t_eval[0]
    dt_scaled = dt * OMEGA_L
    n = 8 * N_T
    # frequency axis for the scaled time (units 1/T_L)
    freqs = np.fft.rfftfreq(n, d=dt_scaled) * 2 * np.pi
    mask = freqs <= MAX_FREQ

    # FFT magnitudes
    fft_sum = np.abs(np.fft.rfft(current, n=n))

    # normalize to max = 1 for each
    if fft_sum.max() != 0:
        fft_sum /= fft_sum.max()

    fig_fft, ax_fft = plt.subplots()
    ax_fft.plot(freqs[mask], fft_sum[mask], label=r'FFT($j_1(t)$)')
    ax_fft.set_xlabel(r'$\omega / \omega_L$')   # frequency in units of laser frequency
    ax_fft.set_ylabel('Normalized FFT magnitude')
    ax_fft.set_xlim(0, freqs.max())
    ax_fft.legend()
    ax_fft.set_yscale("log")

    for i in range(1, MAX_FREQ):
        ax_fft.axvline(i, c="k", ls="--", alpha=0.5)
    ax_fft.set_xlim(0, MAX_FREQ)

    fig_fft.tight_layout()

    plt.show()