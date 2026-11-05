import numpy as np
from scipy.optimize import brentq, root

from Momentum import Gamma, Momentum


_CXX_PRECISION = 2.0**-39


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


def _flow_arrays(flow_data):
    L = flow_data["L"]
    if L % 2:
        raise ValueError("The NickelCUT susceptibility calculation requires even L.")

    n_momenta = L * L
    differing_spin = np.asarray(
        flow_data["interactions_differing_spin"], dtype=float
    ).reshape(n_momenta, n_momenta, n_momenta)
    same_spin = np.asarray(
        flow_data["interactions_same_spin"], dtype=float
    ).reshape(n_momenta, n_momenta, n_momenta)
    epsilon_tilde = np.asarray(flow_data["epsilon_tilde"], dtype=float).reshape(-1)
    dispersion_0 = np.asarray(flow_data["dispersion"], dtype=float).reshape(-1)
    if epsilon_tilde.size != n_momenta or dispersion_0.size != n_momenta:
        raise ValueError("Dispersion arrays must contain exactly L*L values.")

    return L, differing_spin, same_spin, epsilon_tilde, dispersion_0


def _momentum_coordinates(L):
    indices = np.arange(L * L)
    return indices % L, indices // L


def _shifted_positions(L, x, y):
    return (x % L) + L * (y % L)


def _interpolate_periodic_grid(values, L, factor):
    """Bilinearly interpolate values sampled on a periodic L x L grid."""
    values = np.asarray(values)
    if factor == 1:
        return values

    grid = values.reshape((L, L) + values.shape[1:])
    fine_coordinates = np.arange(L * factor, dtype=float) / factor
    lower = np.floor(fine_coordinates).astype(int)
    upper = (lower + 1) % L
    fraction = fine_coordinates - lower
    trailing_dimensions = (1,) * (grid.ndim - 2)
    fraction_x = fraction[:, None].reshape(
        (L * factor, 1) + trailing_dimensions
    )
    fraction_y = fraction[None, :].reshape(
        (1, L * factor) + trailing_dimensions
    )

    lower_lower = grid[lower[:, None], lower[None, :]]
    upper_lower = grid[upper[:, None], lower[None, :]]
    lower_upper = grid[lower[:, None], upper[None, :]]
    upper_upper = grid[upper[:, None], upper[None, :]]
    interpolated = (
        (1.0 - fraction_x) * (1.0 - fraction_y) * lower_lower
        + fraction_x * (1.0 - fraction_y) * upper_lower
        + (1.0 - fraction_x) * fraction_y * lower_upper
        + fraction_x * fraction_y * upper_upper
    )
    return interpolated.reshape(
        (L * factor * L * factor,) + values.shape[1:]
    )


def _interpolate_periodic_interaction_matrix(matrix, L, factor):
    """Interpolate U(k,p) and include the finer-grid momentum-cell weight."""
    if factor == 1:
        return matrix

    row_interpolated = _interpolate_periodic_grid(matrix, L, factor)
    interpolated = _interpolate_periodic_grid(row_interpolated.T, L, factor).T
    return interpolated / factor**2


def _chemical_potential(beta, energies, target_filling):
    if beta <= 0.0:
        raise ValueError("Hartree-Fock filling requires a positive inverse temperature.")

    margin = max(50.0 / beta, 1.0)
    lower = np.min(energies) - margin
    upper = np.max(energies) + margin

    def filling_residual(mu):
        occupations = compute_occupation_numbers(beta, energies - mu)
        return target_filling - 2.0 * np.mean(occupations)

    return brentq(filling_residual, lower, upper, xtol=1e-12)


def _resolve_beta(flow_data, beta):
    if beta is None:
        beta = flow_data.get("beta")
    if beta is None:
        temperature = flow_data.get("T")
        if temperature is None:
            raise ValueError("Pass beta or include beta/T in flow_data.")
        beta = 1.0 / temperature if temperature > 0.0 else 100.0
    elif beta <= 0.0 and flow_data.get("T", 0.0) <= 0.0:
        # C++ stores -1 as the beta sentinel at T=0, then Model maps it to 100.
        beta = 100.0
    if beta <= 0.0:
        raise ValueError("Inverse temperature beta must be positive.")
    return float(beta)


def hartree_fock(flow_data, beta=None, target_filling=None):
    """Return the converged HF dispersion with its chemical potential removed.

    The iteration follows the self-energy contraction and filling constraint
    used by the C++ susceptibility implementation.
    """
    L, differing_spin, same_spin, epsilon_tilde, dispersion_0 = _flow_arrays(flow_data)
    beta = _resolve_beta(flow_data, beta)
    if target_filling is None:
        target_filling = flow_data.get("filling")
    if target_filling is None:
        raise ValueError("Pass target_filling or include it in flow_data.")
    target_filling = float(target_filling)

    k_x, k_y = _momentum_coordinates(L)
    q_minus_p = _shifted_positions(
        L,
        k_x[None, :] - k_x[:, None] + L // 2,
        k_y[None, :] - k_y[:, None] + L // 2,
    )
    gamma = Gamma(L).pos
    p_indices = np.arange(L * L)[:, None]
    q_indices = np.arange(L * L)[None, :]
    hf_interaction = (
        2.0 * differing_spin[:, :, gamma]
        + 2.0 * (
            same_spin[:, :, gamma]
            - same_spin[p_indices, q_indices, q_minus_p]
        )
    )

    def self_consistency_residual(self_energy):
        energies = epsilon_tilde + self_energy
        chemical_potential = _chemical_potential(
            beta, energies, target_filling
        )
        occupations = compute_occupation_numbers(
            beta, energies - chemical_potential
        )
        return self_energy - hf_interaction @ occupations

    initial_self_energy = dispersion_0 - epsilon_tilde
    solution = root(
        self_consistency_residual,
        initial_self_energy,
        method="broyden1",
        options={"maxiter": 600, "fatol": 1e-10},
    )
    residual = self_consistency_residual(solution.x)
    if not np.all(np.isfinite(solution.x)) or np.max(np.abs(residual)) > 1e-10:
        raise RuntimeError(
            "Hartree-Fock self-consistency did not converge: "
            f"residual={np.max(np.abs(residual)):.3e}"
        )

    energies = epsilon_tilde + solution.x
    chemical_potential = _chemical_potential(beta, energies, target_filling)
    return energies - chemical_potential


def _log_cosh(values):
    absolute_values = np.abs(values)
    return (
        absolute_values
        + np.log1p(np.exp(-2.0 * absolute_values))
        - np.log(2.0)
    )


def _log_sinhc(values):
    absolute_values = np.abs(values)
    result = np.empty_like(absolute_values, dtype=float)
    small = absolute_values < 1e-3
    squared = absolute_values[small] ** 2
    result[small] = squared * (
        1.0 / 6.0
        + squared * (-1.0 / 180.0 + squared / 2835.0)
    )
    large = ~small
    result[large] = (
        absolute_values[large]
        + np.log1p(-np.exp(-2.0 * absolute_values[large]))
        - np.log(2.0 * absolute_values[large])
    )
    return result


def _static_chi_diagonal(beta, dispersion, shifted_dispersion, pairing):
    a = 0.5 * beta * dispersion
    b = 0.5 * beta * shifted_dispersion
    argument = a + b if pairing else a - b
    prefactor = 0.25 * beta if pairing else 0.5 * beta
    return np.exp(
        np.log(prefactor)
        + _log_sinhc(argument)
        - _log_cosh(a)
        - _log_cosh(b)
    )


def _static_interaction_matrix(flow_data, channel, transfer):
    L, differing_spin, same_spin, _, _ = _flow_arrays(flow_data)
    k_x, k_y = _momentum_coordinates(L)
    q_x, q_y = transfer.x, transfer.y
    l_plus_q = _shifted_positions(
        L,
        k_x[None, :] + q_x + L // 2,
        k_y[None, :] + q_y + L // 2,
    )
    k_l_minus = _shifted_positions(
        L,
        k_x[None, :] - k_x[:, None] + L // 2,
        k_y[None, :] - k_y[:, None] + L // 2,
    )
    k_indices = np.arange(L * L)[:, None]
    l_indices = np.arange(L * L)[None, :]

    if channel == "SC":
        return 2.0 * differing_spin[k_indices, l_plus_q, transfer.pos]

    differing = differing_spin[k_indices, l_plus_q, transfer.pos]
    same_direct = same_spin[k_indices, l_plus_q, transfer.pos]
    same_exchange = same_spin[k_indices, l_indices, k_l_minus]
    if channel == "spin":
        return -differing + same_direct - same_exchange
    if channel == "charge":
        return differing + same_direct - same_exchange
    raise ValueError(f"Unknown static susceptibility channel: {channel}")


def _high_symmetry_path(L):
    half = L // 2
    return (
        [Momentum(L, half - step, half) for step in range(half + 1)]
        + [Momentum(L, 0, half - step) for step in range(1, half + 1)]
        + [Momentum(L, step, step) for step in range(1, half + 1)]
    )


def _channel_static_susceptibilities(
    flow_data,
    beta,
    dispersion,
    states,
    channel,
    internal_momentum_interpolation_factor,
):
    L = flow_data["L"]
    k_x, k_y = _momentum_coordinates(L)
    results = []
    leading_eigenvalues = []
    found_negative_excitation = False

    def matrices_at(transfer):
        shifted_positions = _shifted_positions(
            L,
            k_x + transfer.x + L // 2,
            k_y + transfer.y + L // 2,
        )
        pairing = channel == "SC"
        chi = _static_chi_diagonal(
            beta, dispersion, dispersion[shifted_positions], pairing
        )
        chi = _interpolate_periodic_grid(
            chi, L, internal_momentum_interpolation_factor
        )
        interaction = _static_interaction_matrix(flow_data, channel, transfer)
        interaction = _interpolate_periodic_interaction_matrix(
            interaction, L, internal_momentum_interpolation_factor
        )
        return chi, interaction

    def append_susceptibility(chi, interaction, transfer):
        nonlocal found_negative_excitation
        sqrt_chi = np.sqrt(chi)
        interaction_norm = np.linalg.norm(interaction)
        if interaction_norm > 0.0:
            hermiticity_error = (
                np.linalg.norm(interaction - interaction.T) / interaction_norm
            )
            if hermiticity_error >= _CXX_PRECISION:
                raise ValueError(
                    f"{channel} interaction matrix is not Hermitian at "
                    f"transfer ({transfer.x}, {transfer.y}): "
                    f"relative error={hermiticity_error:.3e}"
                )
        kernel = sqrt_chi[:, None] * interaction * sqrt_chi[None, :]

        eigenvalues, eigenvectors = np.linalg.eigh(kernel)
        leading_eigenvalues.append(float(eigenvalues[0]))
        shifted_eigenvalues = eigenvalues + 1.0
        small = np.abs(shifted_eigenvalues) < _CXX_PRECISION
        inverse_eigenvalues = np.empty_like(shifted_eigenvalues)
        inverse_eigenvalues[small] = 1.0 / _CXX_PRECISION
        inverse_eigenvalues[~small] = 1.0 / shifted_eigenvalues[~small]
        found_negative_excitation |= np.any(inverse_eigenvalues < 0.0)

        projections = eigenvectors.T @ (sqrt_chi[:, None] * states.T)
        values = np.sum(
            projections**2 * inverse_eigenvalues[:, None],
            axis=0,
        )
        results.append(values.tolist())

    for transfer in _high_symmetry_path(L):
        chi, interaction = matrices_at(transfer)
        append_susceptibility(chi, interaction, transfer)
    return results, leading_eigenvalues, bool(found_negative_excitation)


def compute_static_susceptibilities(
    flow_data,
    beta=None,
    target_filling=None,
    internal_momentum_interpolation_factor=1,
):
    """Compute C++-equivalent static IEOM susceptibilities at high-symmetry q.

    ``flow_data`` must contain L, dispersion, epsilon_tilde, and both raw
    interaction tensors. Beta and filling can be supplied or read from it.
    ``internal_momentum_interpolation_factor`` linearly interpolates the chi and
    interaction matrices over their internal k and p momenta on the periodic
    Brillouin zone. A factor of 1 evaluates the original grid; larger positive
    integer factors increase each internal momentum-grid dimension by that
    factor. Interaction values include the corresponding momentum-cell weight.
    The returned dictionary uses the same keys and channel/state ordering as
    ``ieom_susceptibility.json.gz`` produced by ``susceptibility_ieom``.
    """
    if (
        isinstance(internal_momentum_interpolation_factor, (bool, np.bool_))
        or not isinstance(internal_momentum_interpolation_factor, (int, np.integer))
        or internal_momentum_interpolation_factor < 1
    ):
        raise ValueError(
            "internal_momentum_interpolation_factor must be a positive integer."
        )
    internal_momentum_interpolation_factor = int(
        internal_momentum_interpolation_factor
    )

    L, _, _, _, _ = _flow_arrays(flow_data)
    beta = _resolve_beta(flow_data, beta)
    if target_filling is None:
        target_filling = flow_data.get("filling")
    if target_filling is None:
        raise ValueError("Pass target_filling or include it in flow_data.")
    target_filling = float(target_filling)

    dispersion = hartree_fock(flow_data, beta, target_filling)
    occupations = compute_occupation_numbers(beta, dispersion)

    internal_L = L * internal_momentum_interpolation_factor
    k_x, k_y = _momentum_coordinates(internal_L)
    kx = np.pi * (2.0 * k_x / internal_L - 1.0)
    ky = np.pi * (2.0 * k_y / internal_L - 1.0)
    n_internal_momenta = internal_L * internal_L
    s_wave = np.full(n_internal_momenta, 1.0 / np.sqrt(n_internal_momenta))
    d_wave = np.cos(kx) - np.cos(ky)
    d_wave /= np.linalg.norm(d_wave)
    states = np.stack((s_wave, d_wave))

    result = {"dispersion": dispersion.tolist()}
    result["q_points"] = [
        [transfer.x, transfer.y] for transfer in _high_symmetry_path(L)
    ]
    result["leading_eigenvalues"] = {}
    for channel in ("SC", "spin", "charge"):
        susceptibilities, leading_eigenvalues, negative_excitation = (
            _channel_static_susceptibilities(
                flow_data,
                beta,
                dispersion,
                states,
                channel,
                internal_momentum_interpolation_factor,
            )
        )
        result[channel] = susceptibilities
        result["leading_eigenvalues"][channel] = leading_eigenvalues
        result[f"negative_excitation_{channel.lower()}"] = negative_excitation
    result["hartree_fock_filling"] = float(2.0 * np.mean(occupations))

    metadata = ("time", "T", "U_0", "tprime", "E_F")
    for key in metadata:
        if key in flow_data:
            value = flow_data[key]
            result[key] = value.item() if isinstance(value, np.generic) else value
    result["L"] = int(L)
    result["beta"] = beta
    result["filling"] = target_filling
    result["internal_momentum_interpolation_factor"] = (
        internal_momentum_interpolation_factor
    )
    result["internal_momentum_grid_size"] = internal_L
    return result


def compute_dynamical_matrix_sc(flow_data, occupation_numbers, x):
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


def compute_norm_matrix_sc(L, occupation_numbers, x):
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


def chi_ieom_sc(flow_data, occupation_numbers, x, state):
    L = flow_data["L"]
    norm_matrix = compute_norm_matrix_sc(L, occupation_numbers, x)
    dynamical_matrix = compute_dynamical_matrix_sc(flow_data, occupation_numbers, x)
    tmp = norm_matrix @ state
    return np.dot(tmp.T @ np.linalg.pinv(dynamical_matrix, hermitian=True), tmp) 


def chi_ieom_spin(flow_data, occupation_numbers, x, state):
    L = flow_data["L"]
    norm_matrix = compute_norm_matrix_spin(L, occupation_numbers, x)
    dynamical_matrix = compute_dynamical_matrix_spin(flow_data, occupation_numbers, x)
    tmp = norm_matrix @ state
    return np.dot(tmp.T @ np.linalg.pinv(dynamical_matrix, hermitian=True), tmp) 
