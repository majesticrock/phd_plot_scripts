import numpy as np
import matplotlib.pyplot as plt
import os

from one_atom import *

DELTA = 40
J = -np.sqrt(4. + DELTA)
MU = 0.25 * DELTA - 0.5 * np.sqrt(4 * J**2 + 0.25 * DELTA**2)

def sqrt_expression(k):
    return np.sqrt(4 * J**2 * np.cos(0.5 * np.pi * k)**2 + 0.25 * DELTA**2)

def dispersion_plus(k):
    return -MU + 0.5 * DELTA + sqrt_expression(k)

def dispersion_minus(k):
    return -MU + 0.5 * DELTA - sqrt_expression(k)

def two_cos(k):
    return -2. * np.cos(np.pi * k)

if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set_ylabel("$\\varepsilon$")
    ax.set_xlabel("$k a / \\pi$")

    x = np.linspace(-1, 1, 400)
    ax.plot(x, two_cos(x), label="$2 \\cos k$")
    ax.plot(x, dispersion_plus(x), ls="--", label="Band A")
    ax.plot(x, dispersion_minus(x), ls="-.", label="Band B")
    ax.legend()
    ax.grid()

    if not os.path.isfile("two_atom.dat"):
        plt.show()
        exit()
    
    t_eval, current = np.loadtxt("two_atom.dat")
    
    fig_j, ax_j = plt.subplots()
    ax_j.plot(t_eval * OMEGA_L, current, label="$j_2(t)$")
    ax_j.set_xlabel(r'$t / T_L$')
    ax_j.set_ylabel(r'Signal')
    ax_j.legend()
    fig_j.tight_layout()


    dt = t_eval[1] - t_eval[0]
    dt_scaled = dt * OMEGA_L
    n = 8 * N_T
    freqs = np.fft.rfftfreq(n, d=dt_scaled) * 2 * np.pi
    fft_S = np.abs(np.fft.rfft(current, n=n))
    if fft_S.max() != 0:
        fft_S /= fft_S.max()

    fig_fft, ax_fft = plt.subplots()
    mask = freqs <= MAX_FREQ
    ax_fft.plot(freqs[mask], fft_S[mask], label=r'FFT($j_2(t)$)')
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