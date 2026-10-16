import numpy as np
import matplotlib.pyplot as plt

from Momentum import Momentum
from ieom import (
    chi_ieom as _chi_ieom,
    compute_dynamical_matrix as _compute_dynamical_matrix,
    compute_norm_matrix as _compute_norm_matrix,
    compute_occupation_numbers,
    hartree_fock as _hartree_fock,
)
from load_flow_files import load_final_flow_state
from nickel_cut_parameters import FLOW_PARAMETERS


data = load_final_flow_state(**FLOW_PARAMETERS, resume_num="", force_json=False)
L = data["L"]
N = L * L

def hartree_fock(beta):
    return _hartree_fock(data, beta)


def compute_dynamical_matrix(occupation_numbers, x):
    return _compute_dynamical_matrix(data, occupation_numbers, x)


def compute_norm_matrix(occupation_numbers, x):
    return _compute_norm_matrix(L, occupation_numbers, x)

def chi_ieom(occupation_numbers, x):
    return _chi_ieom(data, occupation_numbers, x)

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


beta, f, vec, weight = find_ieom_divergence(
                    Momentum(L, L//2, L//2),
                    beta_min=0.1,
                    beta_max=30.,
                    beta_tolerance=0.01,
                    weight_tolerance=1e-11)

fig_vector, ax_vector = plt.subplots(layout="constrained")
ax_vector.set_title(f"$T={1./beta:.6f}$  $W={weight:.6f}$")
print(f"$T={1./beta:.6f}$  $W={weight:.6f}$")
im = ax_vector.imshow(vec.reshape(L, L), aspect="equal")
cbar = fig_vector.colorbar(im, ax=ax_vector)
cbar.set_label("Eigenvector")
ax_vector.set_xlabel("$q_x$")
ax_vector.set_xlabel("$q_y$")
beta *= 0.9
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
