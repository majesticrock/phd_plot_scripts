import numpy as np
import matplotlib.pyplot as plt

from ieom import (
    compute_static_susceptibilities,
    compute_dynamical_matrix_spin as _compute_dynamical_matrix_spin,
    compute_norm_matrix_spin as _compute_norm_matrix_spin,
    compute_occupation_numbers,
    hartree_fock as _hartree_fock,
)
from load_flow_files import load_final_flow_state
from nickel_cut_parameters import FLOW_PARAMETERS


INTERNAL_MOMENTUM_INTERPOLATION_FACTOR = 1

data = load_final_flow_state(**FLOW_PARAMETERS, resume_num="", force_json=False)
L = data["L"]
N = L * L

def hartree_fock(beta):
    return _hartree_fock(data, beta)


def compute_dynamical_matrix(occupation_numbers, x):
    return _compute_dynamical_matrix_spin(data, occupation_numbers, x)


def compute_norm_matrix(occupation_numbers, x):
    return _compute_norm_matrix_spin(L, occupation_numbers, x)

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


susceptibility_data = compute_static_susceptibilities(
    data,
    internal_momentum_interpolation_factor=INTERNAL_MOMENTUM_INTERPOLATION_FACTOR,
)
for (q_x, q_y), eigenvalue in zip(
    susceptibility_data["q_points"],
    susceptibility_data["leading_eigenvalues"]["spin"],
):
    print(
        f"spin q=({q_x}, {q_y}): leading eigenvalue={eigenvalue:.10g}, "
        f"1 + eigenvalue={1.0 + eigenvalue:.10g}"
    )
chi_s_values, chi_d_values = np.asarray(susceptibility_data["spin"]).T
n = L // 2

fig, ax = plt.subplots()
ax.plot(np.arange(len(chi_s_values)), chi_s_values, "r-", label="$s$-wave")
ax.plot(np.arange(len(chi_d_values)), chi_d_values, "b--", label="$d$-wave")

ax.set_xticks([0, n, 2 * n, 3 * n])
ax.set_xticklabels([r"$\Gamma$", "X", "M", r"$\Gamma$"])
ax.set_xlabel(r"$\mathbf{x}$")
ax.set_ylabel(r"$\chi_{\mathrm{IEOM}}(\mathbf{x})$")
ax.grid(axis="x", linestyle=":")
ax.legend()
fig.tight_layout()

plt.show()
