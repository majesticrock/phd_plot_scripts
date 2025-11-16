import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------------------
# Parameters
# ---------------------------
Delta = 50.0
J = -15.525293  # assumed precomputed

A0 = 15.
OMEGA_L = 1.
N_LASER = 4.
OMEGA_ENV = OMEGA_L / N_LASER
T_MAX = 2. * N_LASER * np.pi / OMEGA_L
N_K = 200
N_T = 20000

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
    
# W is j(t) / i
W_exp_BZ = np.zeros(len(times), dtype=complex)

# Loop over k
for ik, k0 in enumerate(k_vals):
    # Initial eigenvectors
    eigvals0, eigvecs0 = np.linalg.eigh(H(k0))
    occupied_indices = np.where(eigvals0 < 0)[0]

    psi0 = eigvecs0
    psi0_flat = np.ascontiguousarray(psi0).reshape(-1).view(np.float64)

    # Time evolution for this k
    def schrodinger_all(t, psi_flat):
        psi = psi_flat.view(np.complex128).reshape((3, 3))
        k_eff = k0 - vector_potential(t)
        H_t = H(k_eff)
        dpsi_dt = -1j * H_t @ psi
        return dpsi_dt.reshape(-1).view(np.float64)

    sol = solve_ivp(
        fun=schrodinger_all,
        t_span=[0, T_MAX],
        y0=psi0_flat,
        t_eval=times,
        method='DOP853',
        rtol=1e-8,   # relative tolerance
        atol=1e-12    # absolute tolerance
    )

    psi_t = np.ascontiguousarray(sol.y.T).view(np.complex128)
    psi_t = psi_t.reshape((-1, 3, 3)).transpose(1, 2, 0)

    # Compute T=0 expectation value for this k
    for i, t in enumerate(times):
        k_eff = k0 - vector_potential(t)
        W_t = W(k_eff)
        W_sum_k = 0
        for n in occupied_indices:
            vec = psi_t[:, n, i]
            W_sum_k += np.vdot(vec, W_t @ vec)
        W_exp_BZ[i] += W_sum_k

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