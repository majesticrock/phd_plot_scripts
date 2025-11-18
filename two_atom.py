import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from one_atom import *
from two_atom_overview import *

def sigma_eq_expectation(k):
    Jk = 4 * J * np.cos(k / 2.0)
    h_x, h_z = Jk, -0.5 * DELTA
    h_mag = np.sqrt(h_x**2 + h_z**2)

    E_plus = dispersion_plus(k)
    E_minus = dispersion_minus(k)

    f_plus = fermi_function(E_plus)
    f_minus = fermi_function(E_minus)
    
    pref = (f_plus - f_minus) / h_mag
    return np.array([pref * h_x, 0.0, pref * h_z])

ks = np.linspace(-np.pi, np.pi, N_K, endpoint=False)
t_eval = np.linspace(0.0, T_MAX, N_T)
y0 = np.concatenate([sigma_eq_expectation(k) for k in ks])


def right_side(t, y, k):
    k_eff = k -vector_potential(t)
    equilibrium = sigma_eq_expectation(k_eff)
    Jk = 4. * J * np.cos(0.5 * k_eff)
    
    dx =  DELTA * y[1] + RELAX * (equilibrium[0] - y[0])
    dy = -DELTA * y[0] - 2.0 * Jk * y[2] + RELAX * (equilibrium[1] - y[1])
    dz =  2.0 * Jk * y[1] + RELAX * (equilibrium[2] - y[2])
    
    return np.array([dx, dy, dz])

current = np.zeros_like(t_eval)
for k in ks:
    solution = solve_ivp(right_side, [0, T_MAX], sigma_eq_expectation(k),
                        method="DOP853", t_eval=t_eval, args=(k,),
                        rtol=1e-6, atol=1e-8)
    current += solution.y[1,:] * np.sin(0.5 * (k - vector_potential(t_eval)))
current *= 0.5 * J / N_K

np.savetxt("two_atom.dat", np.array([t_eval, current]))

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