import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

DELTA = 10
J = np.sqrt(4. + DELTA)
MU = 0.75 * DELTA + 0.5 * np.sqrt(4 * J**2 + 0.25 * DELTA**2)

def sqrt_expression(k):
    return np.sqrt(4 * J**2 * np.cos(0.5 * np.pi * k)**2 + 0.25 * DELTA**2)

def dispersion_plus(k):
    return -MU + 0.5 * DELTA + sqrt_expression(k)

def dispersion_minus(k):
    return -MU + 0.5 * DELTA - sqrt_expression(k)

def two_cos(k):
    return 2. * np.cos(np.pi * k)

fig, ax = plt.subplots()
ax.set_ylabel("$\\varepsilon$")
ax.set_xlabel("$k a / \\pi$")


x = np.linspace(-1, 1, 400)
ax.plot(x, two_cos(x), label="$2 \\cos k$")
ax.plot(x, dispersion_plus(x), ls="--", label="Band A")
ax.plot(x, dispersion_minus(x), ls="-.", label="Band B")
ax.legend()

def fermi_function(epsilon, beta=1e2):
    return 1. / (np.exp(beta * epsilon) + 1.)

def sigma_eq_expectation(k, T=1e-2):
    Jk = 4 * J * np.cos(k / 2.0)
    h_x, h_z = Jk, -0.5 * DELTA
    h_mag = np.sqrt(h_x**2 + h_z**2)

    E_plus = dispersion_plus(k)
    E_minus = dispersion_minus(k)

    beta = 1.0 / T
    f_plus = fermi_function(E_plus, beta)
    f_minus = fermi_function(E_minus, beta)
    
    pref = (f_plus - f_minus) / h_mag
    return np.array([pref * h_x, 0.0, pref * h_z])

A0 = 1.5
OMEGA_L = 1.
N_LASER = 8
OMEGA_ENV = OMEGA_L / N_LASER
T_MAX = 2 * N_LASER * np.pi / OMEGA_L
N_K = 200
N_T = 2000

ks = np.linspace(-np.pi, np.pi, N_K, endpoint=False)
t_eval = np.linspace(0.0, T_MAX, N_T)
y0 = np.concatenate([sigma_eq_expectation(k) for k in ks])


def vector_potential(t):
    return 0.5 * A0 * np.cos(OMEGA_L * t) * (1. - np.cos(OMEGA_ENV * t))

def right_side_all(t, y, ks):
    n = len(ks)
    y = y.reshape(n, 3)
    sigx, sigy, sigz = y[:,0], y[:,1], y[:,2]

    Ak = vector_potential(t)
    Jk = 4 * J * np.cos(0.5 * (ks - Ak))

    dx =  DELTA * sigy
    dy = -DELTA * sigx - 2.0 * Jk * sigz
    dz =  2.0 * Jk * sigy

    return np.column_stack([dx, dy, dz]).ravel()

def observable_S(t, y, ks):
    y = y.reshape(len(ks), 3)
    sigy = y[:,1]
    Ak = vector_potential(t)
    weight = np.sin(0.5 * (ks - Ak))
    return np.sum(weight * sigy)

sol = solve_ivp(right_side_all, [0, T_MAX], y0, t_eval=t_eval, args=(ks,), vectorized=True, method="LSODA")

# ----- compute observable -----
S_t = np.zeros_like(t_eval)
for i, t in enumerate(t_eval):
    y = sol.y[:, i].reshape(len(ks), 3)
    sigy = y[:,1]
    weight = np.sin(0.5 * (ks - vector_potential(t)))
    S_t[i] = 0.5 * J * np.sum(weight * sigy) / N_K


fig_j, ax_j = plt.subplots()

ax_j.plot(t_eval * OMEGA_L, S_t / np.max(S_t), label="$j_2(t)$")


A_t = vector_potential(t_eval)  # shape (N_T,)
# Use broadcasting to compute sin(k - A(t)) for all k and t
# ks has shape (N_K,), A_t has shape (N_T,)
# result of (ks[None, :] - A_t[:, None]) has shape (N_T, N_K)
sin_matrix = np.sin(ks[None, :] - A_t[:, None])
f_k = fermi_function(2 * np.cos(ks))  # shape (N_K,)

# Multiply and sum over k (axis=1)
sum_k_t = np.sum(sin_matrix * f_k[None, :], axis=1)
sum_k_t /= N_K

ax_j.plot(t_eval * OMEGA_L, sum_k_t / np.max(sum_k_t), label="$j_1(t)$")

ax_j.set_xlabel(r'$t / T_L$')
ax_j.set_ylabel(r'Signal')
ax_j.legend()

fig_j.tight_layout()


dt = t_eval[1] - t_eval[0]
dt_scaled = dt * OMEGA_L
n = 8 * N_T
# frequency axis for the scaled time (units 1/T_L)
freqs = np.fft.rfftfreq(n, d=dt_scaled) * 2 * np.pi

# FFT magnitudes
fft_S = np.abs(np.fft.rfft(S_t, n=n))
fft_sum = np.abs(np.fft.rfft(sum_k_t, n=n))

# normalize to max = 1 for each
if fft_S.max() != 0:
    fft_S /= fft_S.max()
if fft_sum.max() != 0:
    fft_sum /= fft_sum.max()

fig_fft, ax_fft = plt.subplots()
ax_fft.plot(freqs, fft_S, label=r'FFT($j_2(t)$)')
ax_fft.plot(freqs, fft_sum, label=r'FFT($j_1(t)$)', ls='--')
ax_fft.set_xlabel(r'$\omega / \omega_L$')   # frequency in units of laser frequency
ax_fft.set_ylabel('Normalized FFT magnitude')
ax_fft.set_xlim(0, freqs.max())
ax_fft.legend()
ax_fft.set_yscale("log")

MAX_FREQ = 20

for i in range(1, MAX_FREQ):
    ax_fft.axvline(i, c="k", ls="--", alpha=0.5)
ax_fft.set_xlim(0, MAX_FREQ)

fig_fft.tight_layout()


plt.show()