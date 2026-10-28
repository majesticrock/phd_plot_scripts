import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


def solve_hartree_fock(flow_point, beta, target_filling,
                       max_iter=1200, tol=1e-10, mixing=0.5):
    """
    Solve the spin-invariant Hartree-Fock equations for one flow step.

    Parameters
    ----------
    flow_point : dict
        One entry of data["extracted_channels"].
    beta : float
        Inverse temperature.
    target_filling : float
        Target filling, including both spin species.
    max_iter : int
        Maximum number of self-consistency iterations.
    tol : float
        Self-consistency tolerance.
    mixing : float
        Linear mixing parameter for the self-energy.

    Returns
    -------
    hf_dispersion : ndarray
        Converged dispersion, shifted by the chemical potential.
    mu : float
        Self-consistent chemical potential.
    self_energy : ndarray
        Converged Hartree-Fock self-energy.
    """
    epsilon = np.asarray(
        flow_point["epsilon_tilde"], dtype=float
    ).reshape(-1)

    raw_dispersion = np.asarray(
        flow_point["dispersion"], dtype=float
    ).reshape(-1)

    n_k = len(epsilon)

    differing = np.asarray(
        flow_point["single_particle_energy_differing"],
        dtype=float,
    ).reshape(n_k, n_k)

    same = np.asarray(
        flow_point["single_particle_energy_same"],
        dtype=float,
    ).reshape(n_k, n_k)

    # The interaction normalization is assumed to match the C++ code.
    interaction = differing + same

    # Initialize as in the C++ implementation.
    sigma = raw_dispersion - epsilon
    
    def chemical_potential(energies):
        # The occupation is per spin, so the total filling has a factor 2.
        def filling_residual(mu):
            occupations = expit(-beta * (energies - mu))
            filling = 2.0 * np.mean(occupations)
            return target_filling - filling

        margin = max(50.0 / beta, 1.0)
        lower = np.min(energies) - margin
        upper = np.max(energies) + margin

        return brentq(
            filling_residual, lower, upper,
            xtol=1e-12,
        )

    for iteration in range(max_iter):
        energies = epsilon + sigma
        mu = chemical_potential(energies)

        occupations = expit(-beta * (energies - mu))

        sigma_raw = interaction @ occupations

        # Linear mixing, as in the C++ implementation.
        sigma_new = (
            (1.0 - mixing) * sigma + mixing * sigma_raw
        )

        residual = np.max(np.abs(sigma_new - sigma))
        sigma = sigma_new

        if residual < tol:
            break
    else:
        raise RuntimeError(
            f"Hartree-Fock did not converge after {max_iter} "
            f"iterations; final residual = {residual:.3e}"
        )

    # Recompute the chemical potential and final dispersion.
    energies = epsilon + sigma
    mu = chemical_potential(energies)

    hf_dispersion = energies - mu

    return hf_dispersion, mu, sigma