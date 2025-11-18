import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from one_atom import *

# ---------------------------
# Parameters
# ---------------------------
Delta = 50.0
J = -15.525293  # assumed precomputed

k_vals = np.linspace(-np.pi, np.pi, N_K, endpoint=False)
times = np.linspace(0, T_MAX, N_T)

def vector_potential(t):
    return 0.5 * A0 * np.cos(OMEGA_L * t) * (1. - np.cos(OMEGA_ENV * t))

# Hamiltonian H(k_eff)
def H(k_eff):
    omega = np.exp(1j * k_eff / 3)
    return np.array([
        [Delta, J * omega, J * np.conj(omega)],
        [J * np.conj(omega), 0, J * omega],
        [J * omega, J * np.conj(omega), Delta]
    ], dtype=complex)

# W(k_eff) matrix
def W(k_eff):
    w = np.exp(1j * k_eff / 3)
    w_conj = np.conj(w)
    return np.array([
        [0, w, -w_conj],
        [-w_conj, 0, w],
        [w, -w_conj, 0]
    ], dtype=complex)

def rho(k_eff):
    eigvals, eigvecs = np.linalg.eigh(H(k_eff))
    boltz = np.exp(-BETA * eigvals)
    Z = np.sum(boltz)
    return eigvecs @ np.diag(boltz / Z) @ eigvecs.conj().T
    
    
# W is j(t) / i
W_exp_BZ = np.zeros(len(times), dtype=complex)

for ik, k0 in enumerate(k_vals):

    # Initial thermal density matrix
    rho0 = rho(k0)
    rho0_flat = np.ascontiguousarray(rho0).reshape(-1).view(np.float64)

    # Liouville-von Neumann evolution: dρ/dt = -i[H,ρ]
    def liouville(t, rho_flat):
        __rho = rho_flat.view(np.complex128).reshape((3, 3))
        k_eff = k0 - vector_potential(t)
        H_t = H(k_eff)
        comm = H_t @ __rho - __rho @ H_t
        return (-1j * comm + RELAX * (rho(k_eff) - __rho)).reshape(-1).view(np.float64)

    # Solve for ρ(t)
    sol = solve_ivp(
        fun=liouville,
        t_span=[0, T_MAX],
        y0=rho0_flat,
        t_eval=times,
        method='DOP853',
        rtol=1e-6,
        atol=1e-8
    )

    rho_t = np.ascontiguousarray(sol.y.T).view(np.complex128)
    rho_t = rho_t.reshape((-1, 3, 3)).transpose(1, 2, 0)

    # Expectation value from density matrix: Tr[ρ(t) W(t)]
    for i, t in enumerate(times):
        k_eff = k0 - vector_potential(t)
        W_t = W(k_eff)
        rho_now = rho_t[:, :, i]
        W_exp_BZ[i] += np.trace(rho_now @ W_t)


# Normalize by number of k-points (optional)
W_exp_BZ *= 1j * J / (3 * N_K)

# Plot
fig, ax = plt.subplots()
ax.plot(times, W_exp_BZ.imag, label=r'imag')
ax.plot(times, W_exp_BZ.real, label=r'real')
ax.set_xlabel('$t$')
ax.set_ylabel('$j(t)$')
ax.legend()
ax.grid(True)

###############################
#psi_mag = np.abs(psi_t)          # modulus of each complex entry
#psi_norms = np.linalg.norm(psi_mag, axis=0)  # shape (3 eigenvectors, N_t) - norm per eigenvector
#
#fig_psi, ax1 = plt.subplots()
#
## top: norm of each eigenvector over time (should be ~1 if evolution is unitary)
#for n in range(psi_norms.shape[0]):
#    ax1.plot(times, psi_norms[n, :], label=f'eigvec {n}')
#ax1.set_ylabel(r'$\|\psi_n(t)\|$')
#ax1.legend(loc='upper right')
#ax1.grid(True)
#
#fig_psi.tight_layout()
###################################


dt = times[1] - times[0]
dt_scaled = dt * OMEGA_L
n = 8 * N_T
# frequency axis for the scaled time (units 1/T_L)
freqs = np.fft.rfftfreq(n, d=dt_scaled) * 2 * np.pi

# FFT magnitudes
fft_j = np.abs(np.fft.rfft(W_exp_BZ.real, n=n))

# normalize to max = 1 for each
if fft_j.max() != 0:
    fft_j /= fft_j.max()

fig_fft, ax_fft = plt.subplots()
ax_fft.plot(freqs, fft_j, label=r'FFT($j_3(t)$)')
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