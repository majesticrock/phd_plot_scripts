import matplotlib.pyplot as plt
import numpy as np

from finite_size_energy_gap import (
    FERMI_LEVEL,
    T_PRIME,
    _build_gap_context,
    _dispersion,
    _find_fermi_surface_mask,
)

L=14

def _contour_points(context, fermi_level):
    segments = context["contour_generator"].lines(fermi_level)
    if segments:
        return np.concatenate(segments)

    close_points = np.isclose(
        context["fine_dispersion"],
        fermi_level,
        rtol=0.0,
        atol=1e-12,
    )
    if not np.any(close_points):
        close_points.flat[
            np.argmin(np.abs(context["fine_dispersion"] - fermi_level))
        ] = True
    return np.column_stack(
        (
            context["fine_kx"][close_points],
            context["fine_ky"][close_points],
        )
    )


def _near_fermi_surface(kx, ky, contour_points, maximum_distance):
    momenta = np.column_stack((kx.ravel(), ky.ravel()))
    differences = momenta[:, np.newaxis, :] - contour_points[np.newaxis, :, :]
    differences = (differences + np.pi) % (2.0 * np.pi) - np.pi
    distances = np.sqrt(np.min(np.sum(differences**2, axis=2), axis=1))
    return (distances <= maximum_distance).reshape(kx.shape)


def adaptive_grid(lattice_size, t_prime, fermi_level):
    """Keep fine samples near the coarse-grid Fermi surface and coarse samples elsewhere."""
    if lattice_size < 4 or lattice_size % 2:
        raise ValueError("L must be an even integer greater than or equal to 4.")

    coarse_size = lattice_size // 2
    coarse_context = _build_gap_context(coarse_size, t_prime)
    coarse_surface_mask = _find_fermi_surface_mask(
        coarse_context, fermi_level
    )
    contour_points = _contour_points(coarse_context, fermi_level)
    maximum_distance = 0.99 * np.pi * np.sqrt(2.0) / coarse_size

    coarse_k = coarse_context["k"]
    coarse_kx, coarse_ky = coarse_context["kx"], coarse_context["ky"]
    coarse_keep = ~coarse_surface_mask

    fine_k = np.linspace(-np.pi, np.pi, lattice_size, endpoint=False)
    fine_kx, fine_ky = np.meshgrid(fine_k, fine_k)
    fine_surface_mask = _near_fermi_surface(
        fine_kx, fine_ky, contour_points, maximum_distance
    )

    coarse_points = np.column_stack(
        (coarse_kx[coarse_keep], coarse_ky[coarse_keep])
    )
    fine_points = np.column_stack(
        (fine_kx[fine_surface_mask], fine_ky[fine_surface_mask])
    )
    coarse_energies = (
        coarse_context["dispersion"][coarse_keep] - fermi_level
    )
    fine_energies = (
        _dispersion(fine_kx, fine_ky, t_prime)[fine_surface_mask] - fermi_level
    )

    return {
        "coarse_size": coarse_size,
        "coarse_context": coarse_context,
        "coarse_surface_mask": coarse_surface_mask,
        "coarse_points": coarse_points,
        "coarse_energies": coarse_energies,
        "fine_kx": fine_kx,
        "fine_ky": fine_ky,
        "fine_surface_mask": fine_surface_mask,
        "fine_points": fine_points,
        "fine_energies": fine_energies,
        "contour_points": contour_points,
        "maximum_distance": maximum_distance,
    }


def main():
    grid = adaptive_grid(L, T_PRIME, FERMI_LEVEL)
    coarse_context = grid["coarse_context"]

    n_adaptive = len(grid['fine_points']) + len(grid['coarse_points'])
    n_uniform = L * L
    print(f"L = {L}, coarse grid = {grid['coarse_size']} x {grid['coarse_size']}")
    print(f"t' = {T_PRIME:g}, Fermi level = {FERMI_LEVEL:g}")
    print(f"Coarse-grid Fermi-surface points: {np.count_nonzero(grid['coarse_surface_mask'])}")
    print(f"Retained fine-grid points near the Fermi surface: {len(grid['fine_points'])}")
    print(f"Retained coarse-grid points away from the Fermi surface: {len(grid['coarse_points'])}")
    print(f"Total adaptive points: {n_adaptive}")
    print(f"Uniform L x L points: {n_uniform}")
    print(f"\nInteraction right-hand-side momentum evaluations:")
    print(f"Adaptive: {n_adaptive * n_adaptive * n_uniform * n_uniform}     Uniform: {n_uniform**4}")

    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    fine = ax.scatter(
        grid["fine_points"][:, 0],
        grid["fine_points"][:, 1],
        c=grid["fine_energies"],
        cmap="coolwarm",
        marker=".",
        s=22,
        label=f"Fine grid ({L} x {L}), near Fermi surface",
    )
    ax.scatter(
        grid["coarse_points"][:, 0],
        grid["coarse_points"][:, 1],
        c=grid["coarse_energies"],
        cmap="coolwarm",
        marker="s",
        s=45,
        edgecolors="black",
        linewidths=0.35,
        label=f"Coarse grid ({grid['coarse_size']} x {grid['coarse_size']}), away from surface",
    )
    if (
        np.min(coarse_context["fine_dispersion"])
        < FERMI_LEVEL
        < np.max(coarse_context["fine_dispersion"])
    ):
        ax.contour(
            coarse_context["fine_kx"],
            coarse_context["fine_ky"],
            coarse_context["fine_dispersion"],
            levels=[FERMI_LEVEL],
            colors="black",
            linewidths=1.0,
        )
    fig.colorbar(fine, ax=ax, label=r"$\varepsilon(\mathbf{k}) - E_F$")
    ax.set(
        xlim=(-np.pi, np.pi),
        ylim=(-np.pi, np.pi),
        aspect="equal",
        xlabel=r"$k_x$",
        ylabel=r"$k_y$",
        title="Adaptive momentum grid around the Fermi surface",
    )
    ax.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
