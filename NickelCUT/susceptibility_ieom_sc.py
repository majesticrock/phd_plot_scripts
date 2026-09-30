import numpy as np
import matplotlib.pyplot as plt

from Momentum import Gamma, Momentum
from load_full_flow_file import load_full_flow_state
from nickel_cut_parameters import FLOW_PARAMETERS


data = load_full_flow_state(**FLOW_PARAMETERS, resume_num="", force_json=False)
L = data["L"]
N = L * L

differing_spin = data["interactions_differing_spin"]
same_spin = data["interactions_same_spin"]
epsilon_tilde = data["epsilon_tilde"]
dispersion_0 = data["dispersion"]

def hartree_fock(beta):
    self_energy = dispersion_0 - epsilon_tilde
    new_self_energy = np.zeros_like(self_energy)
    error = 1000.
    
    while error > 1e-6:
        occupations = compute_occupation_numbers(beta, self_energy + epsilon_tilde)
        new_self_energy = 2. * np.sum((differing_spin[:,:,Momentum(L, L//2, L//2).pos]
                              + 2. * same_spin[:,:, Momentum(L,L//2,L//2).pos]) * occupations[None,:], axis=-1)
        
        error = np.sqrt(np.sum((self_energy - new_self_energy)**2))
        self_energy = 0.5 * (self_energy + new_self_energy)
        
    return self_energy + epsilon_tilde


def compute_occupation_numbers(beta, dispersion):
    """Compute occupations from the stored dispersion at inverse temperature beta."""
    if beta >= 0.0:
        exponent = np.clip(beta * dispersion, -100., 100.)
        return 1.0 / (1.0 + np.exp(exponent))

    occupation_numbers = np.zeros_like(dispersion, dtype=float)
    zero_energy = np.isclose(dispersion, 0.0, atol=1e-10, rtol=0.0)
    occupation_numbers[dispersion < 0.0] = 1.0
    occupation_numbers[zero_energy] = 0.5
    return occupation_numbers


def compute_dynamical_matrix(occupation_numbers, x):
    """Construct the C++ IEOM dynamical matrix for transfer momentum x."""
    dynamical_matrix = np.zeros((N, N), dtype=float)
    gamma = Gamma(L)

    for k_y in range(L):
        for k_x in range(L):
            k = Momentum(L, k_x, k_y)
            kx = (k + x).pos
            diagonal = 0.5 * (epsilon_tilde[k.pos] + epsilon_tilde[kx])

            for K_y in range(L):
                for K_x in range(L):
                    K = Momentum(L, K_x, K_y)
                    diagonal += (
                        differing_spin[kx, K.pos, gamma.pos]
                        + differing_spin[k.pos, K.pos, gamma.pos]
                        - same_spin[kx, K.pos, (K - k - x).pos]
                        + same_spin[kx, K.pos, gamma.pos]
                        - same_spin[k.pos, K.pos, (K - k).pos]
                        + same_spin[k.pos, K.pos, gamma.pos]
                    ) * occupation_numbers[K.pos]

            row_factor = 1.0 - occupation_numbers[k.pos] - occupation_numbers[kx]
            dynamical_matrix[k.pos, k.pos] = 2.0 * row_factor * diagonal

            for l_y in range(L):
                for l_x in range(L):
                    l = Momentum(L, l_x, l_y)
                    col_factor = (
                        1.0 - occupation_numbers[l.pos] - occupation_numbers[(l + x).pos]
                    )
                    dynamical_matrix[k.pos, l.pos] += (
                        2.0 * differing_spin[(-k - x).pos, k.pos, (k - l).pos]
                        * row_factor
                        * col_factor
                    )

    return dynamical_matrix


def compute_norm_matrix(occupation_numbers, x):
    """Construct the diagonal IEOM norm matrix for transfer momentum x."""
    norm_diagonal = np.empty(N, dtype=float)

    for y in range(L):
        for x_index in range(L):
            k = Momentum(L, x_index, y)
            norm_diagonal[k.pos] = (
                occupation_numbers[k.pos]
                + occupation_numbers[(k + x).pos]
                - 1.0
            )

    return np.diag(norm_diagonal)

def chi_ieom(occupation_numbers, x):
    N_mat = compute_norm_matrix(occupation_numbers, x)
    M_mat = compute_dynamical_matrix(occupation_numbers, x)
    return np.sum(N_mat @ np.linalg.pinv(M_mat, hermitian=True) @ N_mat) / N

def find_ieom_divergence(
    x,
    beta_min=0.05,
    beta_max=20.0,
    beta_tolerance=1e-5,
    weight_tolerance=1e-8
):
    """Find the first active zero crossing of an IEOM eigenvalue as beta rises."""
    if beta_min <= 0.0 or beta_max <= beta_min:
        raise ValueError("Require 0 < beta_min < beta_max")
    print(
        f"Searching for an IEOM divergence at x=({x.x}, {x.y}) "
        f"from beta={beta_min:g} to {beta_max:g}"
    )

    def diagonalize(beta):
        dispersion = hartree_fock(beta)
        occupations = compute_occupation_numbers(beta, dispersion)
        norm_diagonal = np.diag(compute_norm_matrix(occupations, x))
        dynamical_matrix = compute_dynamical_matrix(occupations, x)
        eigenvalues, eigenvectors = np.linalg.eigh(dynamical_matrix)

        min_weight = np.sum((norm_diagonal * eigenvectors[:,0])**2) / N
        i=0
        while np.abs(min_weight) < weight_tolerance:
            i+=1
            min_weight = np.sum((norm_diagonal * eigenvectors[:,i])**2) / N
        
        min_value = eigenvalues[i]
        min_vector = norm_diagonal * eigenvectors[:,i]
    
        return min_value, min_vector, min_weight

    f_min, vec_min, weight_min = diagonalize(beta_min)
    f_max, vec_max, weight_max = diagonalize(beta_max)

    # We expect an instability for small T (large beta)
    if (f_max > 0. or f_min < 0):
        print(f_max, f_min, weight_max, weight_min)
        raise ValueError("No root in given interval.")

    while (beta_max - beta_min > beta_tolerance):
        beta_center = 0.5 * (beta_max + beta_min)
        f_center, vec_center, weight_center = diagonalize(beta_center)
        if f_center <= 0.:
            beta_max = beta_center
            f_max = f_center
            vec_max = vec_center
            weight_max = weight_center
        else:
            beta_min = beta_center
            f_min = f_center
            vec_min = vec_center
            weight_min = weight_center

    return beta_center, f_center, vec_center, weight_center


n = L // 2
path = (
    [Momentum(L, n, y) for y in range(n, -1, -1)]
    + [Momentum(L, x, 0) for x in range(n - 1, -1, -1)]
    + [Momentum(L, index, index) for index in range(1, n + 1)]
)

#fig_hf, ax_hf = plt.subplots(layout="constrained")
#hf_dispersion = hartree_fock(10.)
#eps0 = np.asarray([
#    dispersion_0[momentum.pos] for momentum in path
#])
#eps_hf = np.asarray([
#    hf_dispersion[momentum.pos] for momentum in path
#])
#
#ax_hf.plot(np.arange(len(path)), eps0, label="Raw")
#ax_hf.plot(np.arange(len(path)), eps_hf, label="HF")
#ax_hf.set_xticks([0, n, 2 * n, 3 * n])
#ax_hf.set_xticklabels([r"$\Gamma$", "X", "M", r"$\Gamma$"])
#ax_hf.set_xlabel(r"$\mathbf{k}$")
#ax_hf.set_ylabel(r"$\varepsilon_k$")
#ax_hf.grid(axis="x", linestyle=":")
#ax_hf.legend()
#
#plt.show()


#beta, f, vec, weight = find_ieom_divergence(
#                    Momentum(L, L//2, L//2),
#                    beta_min=0.1,
#                    beta_max=40.,
#                    beta_tolerance=0.01,
#                    weight_tolerance=1e-11)
#
#fig_vector, ax_vector = plt.subplots(layout="constrained")
#ax_vector.set_title(f"$T={1./beta:.6f}$  $W={weight:.6f}$")
#print(f"$T={1./beta:.6f}$  $W={weight:.6f}$")
#im = ax_vector.imshow(vec.reshape(L, L), aspect="equal")
#cbar = fig_vector.colorbar(im, ax=ax_vector)
#cbar.set_label("Eigenvector")
#ax_vector.set_xlabel("$q_x$")
#ax_vector.set_xlabel("$q_y$")
#beta *= 0.9
beta = 20.
occupation_numbers = compute_occupation_numbers(beta, hartree_fock(beta))

chi_values = np.asarray([
    chi_ieom(occupation_numbers, momentum) for momentum in path
])

fig, ax = plt.subplots()
ax.plot(np.arange(len(path)), chi_values)
ax.set_xticks([0, n, 2 * n, 3 * n])
ax.set_xticklabels([r"$\Gamma$", "X", "M", r"$\Gamma$"])
ax.set_xlabel(r"$\mathbf{x}$")
ax.set_ylabel(r"$\chi_{\mathrm{IEOM}}(\mathbf{x})$")
ax.grid(axis="x", linestyle=":")
fig.tight_layout()

plt.show()
