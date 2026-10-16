import numpy as np

from Momentum import Gamma, Momentum


def compute_occupation_numbers(beta, dispersion):
    """Compute occupations from the stored dispersion at inverse temperature beta."""
    if beta >= 0.0:
        exponent = np.clip(beta * dispersion, -100.0, 100.0)
        return 1.0 / (1.0 + np.exp(exponent))

    occupation_numbers = np.zeros_like(dispersion, dtype=float)
    zero_energy = np.isclose(dispersion, 0.0, atol=1e-10, rtol=0.0)
    occupation_numbers[dispersion < 0.0] = 1.0
    occupation_numbers[zero_energy] = 0.5
    return occupation_numbers


def hartree_fock(flow_data, beta):
    """Return the self-consistent Hartree-Fock dispersion for a flow snapshot."""
    L = flow_data["L"]
    differing_spin = flow_data["interactions_differing_spin"]
    same_spin = flow_data["interactions_same_spin"]
    epsilon_tilde = flow_data["epsilon_tilde"]
    dispersion_0 = flow_data["dispersion"]

    self_energy = dispersion_0 - epsilon_tilde
    gamma = Gamma(L)
    for _ in range(1000):
        occupations = compute_occupation_numbers(beta, self_energy + epsilon_tilde)
        new_self_energy = 2.0 * np.sum(
            (
                differing_spin[:, :, gamma.pos]
                + 2.0 * same_spin[:, :, gamma.pos]
            )
            * occupations[None, :],
            axis=-1,
        )

        error = np.sqrt(np.sum((self_energy - new_self_energy) ** 2))
        self_energy = 0.5 * (self_energy + new_self_energy)
        if not np.isfinite(error):
            raise RuntimeError("Hartree-Fock self-consistency produced a non-finite error.")
        if error <= 1e-6:
            break
    else:
        raise RuntimeError("Hartree-Fock self-consistency did not converge.")

    return self_energy + epsilon_tilde


def compute_dynamical_matrix(flow_data, occupation_numbers, x):
    """Construct the C++ IEOM dynamical matrix for transfer momentum x."""
    L = flow_data["L"]
    N = L * L
    differing_spin = flow_data["interactions_differing_spin"]
    same_spin = flow_data["interactions_same_spin"]
    epsilon_tilde = flow_data["epsilon_tilde"]
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
                        2.0
                        * differing_spin[(-k - x).pos, k.pos, (k - l).pos]
                        * row_factor
                        * col_factor
                    )

    return dynamical_matrix


def compute_dynamical_matrix_spin(flow_data, occupation_numbers, x):
    """Construct the spin IEOM dynamical matrix for transfer momentum x."""
    L = flow_data["L"]
    N = L * L
    differing_spin = flow_data["interactions_differing_spin"]
    same_spin = flow_data["interactions_same_spin"]
    epsilon_tilde = flow_data["epsilon_tilde"]
    dynamical_matrix = np.zeros((N, N), dtype=float)
    gamma = Gamma(L)

    for k_y in range(L):
        for k_x in range(L):
            k = Momentum(L, k_x, k_y)
            kx = (k + x).pos
            diagonal = 0.5 * (epsilon_tilde[kx] - epsilon_tilde[k.pos])

            for K_y in range(L):
                for K_x in range(L):
                    K = Momentum(L, K_x, K_y)
                    diagonal += (
                        differing_spin[kx, K.pos, gamma.pos]
                        - differing_spin[k.pos, K.pos, gamma.pos]
                        - same_spin[kx, K.pos, (K - k - x).pos]
                        + same_spin[kx, K.pos, gamma.pos]
                        + same_spin[k.pos, K.pos, (K - k).pos]
                        - same_spin[k.pos, K.pos, gamma.pos]
                    ) * occupation_numbers[K.pos]

            row_factor = 2.0 * (
                occupation_numbers[k.pos] - occupation_numbers[kx]
            )
            dynamical_matrix[k.pos, k.pos] = 2.0 * row_factor * diagonal

            for l_y in range(L):
                for l_x in range(L):
                    l = Momentum(L, l_x, l_y)
                    col_factor = 2.0 * (
                        occupation_numbers[l.pos]
                        - occupation_numbers[(l + x).pos]
                    )
                    dynamical_matrix[k.pos, l.pos] += (
                        differing_spin[k.pos, (l + x).pos, x.pos]
                        + same_spin[k.pos, (l + x).pos, x.pos]
                        - same_spin[(-k).pos, (-l - x).pos, (k - l).pos]
                    ) * row_factor * col_factor

    return dynamical_matrix


def compute_norm_matrix(L, occupation_numbers, x):
    """Construct the diagonal IEOM norm matrix for transfer momentum x."""
    norm_diagonal = np.empty(L * L, dtype=float)

    for y in range(L):
        for x_index in range(L):
            k = Momentum(L, x_index, y)
            norm_diagonal[k.pos] = (
                occupation_numbers[k.pos]
                + occupation_numbers[(k + x).pos]
                - 1.0
            )

    return np.diag(norm_diagonal)


def compute_norm_matrix_spin(L, occupation_numbers, x):
    """Construct the diagonal spin IEOM norm matrix for transfer momentum x."""
    norm_diagonal = np.empty(L * L, dtype=float)

    for y in range(L):
        for x_index in range(L):
            k = Momentum(L, x_index, y)
            norm_diagonal[k.pos] = 2.0 * (
                occupation_numbers[k.pos] - occupation_numbers[(k + x).pos]
            )

    return np.diag(norm_diagonal)


def chi_ieom(flow_data, occupation_numbers, x):
    L = flow_data["L"]
    norm_matrix = compute_norm_matrix(L, occupation_numbers, x)
    dynamical_matrix = compute_dynamical_matrix(flow_data, occupation_numbers, x)
    return np.sum(
        norm_matrix
        @ np.linalg.pinv(dynamical_matrix, hermitian=True)
        @ norm_matrix
    ) / (L * L)


def chi_ieom_spin(flow_data, occupation_numbers, x):
    L = flow_data["L"]
    norm_matrix = compute_norm_matrix_spin(L, occupation_numbers, x)
    dynamical_matrix = compute_dynamical_matrix_spin(
        flow_data,
        occupation_numbers,
        x,
    )
    return np.sum(
        norm_matrix
        @ np.linalg.pinv(dynamical_matrix, hermitian=True)
        @ norm_matrix
    ) / (L * L)
