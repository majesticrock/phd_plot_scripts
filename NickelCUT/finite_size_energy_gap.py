import contourpy
import matplotlib.pyplot as plt
import numpy as np


L = 12
T_PRIME = -0.25
FERMI_LEVEL = -0.75
ENERGY_TOLERANCE = 1e-12
FINE_MESH_SIZE = 201


def _dispersion(kx, ky, t_prime):
    return -2.0 * (
        np.cos(kx) + np.cos(ky) + 2.0 * t_prime * np.cos(kx) * np.cos(ky)
    )


def _build_gap_context(lattice_size, t_prime, fine_mesh_size=FINE_MESH_SIZE):
    if lattice_size < 2:
        raise ValueError("L must be at least 2 to have a finite-size energy gap.")
    if fine_mesh_size < 2:
        raise ValueError("The fine mesh must contain at least two points per axis.")

    k = np.linspace(-np.pi, np.pi, lattice_size, endpoint=False)
    kx, ky = np.meshgrid(k, k)
    dispersion = _dispersion(kx, ky, t_prime)
    fine_k = np.linspace(-np.pi, np.pi, fine_mesh_size)
    fine_kx, fine_ky = np.meshgrid(fine_k, fine_k)
    fine_dispersion = _dispersion(fine_kx, fine_ky, t_prime)
    contour_generator = contourpy.contour_generator(
        x=fine_k,
        y=fine_k,
        z=fine_dispersion,
    )

    return {
        "L": lattice_size,
        "T_PRIME": t_prime,
        "k": k,
        "kx": kx,
        "ky": ky,
        "dispersion": dispersion,
        "fine_k": fine_k,
        "fine_kx": fine_kx,
        "fine_ky": fine_ky,
        "fine_dispersion": fine_dispersion,
        "contour_generator": contour_generator,
        "finite_momenta": np.column_stack((kx.ravel(), ky.ravel())),
        "finite_energies": dispersion.ravel(),
    }


def _find_fermi_surface_mask(context, fermi_level):
    fine_dispersion = context["fine_dispersion"]
    if fermi_level < np.min(fine_dispersion) or fermi_level > np.max(fine_dispersion):
        raise ValueError(
            f"Fermi level {fermi_level:g} is outside the dispersion bandwidth "
            f"[{np.min(fine_dispersion):g}, {np.max(fine_dispersion):g}]."
        )

    segments = context["contour_generator"].lines(fermi_level)
    if segments:
        fermi_surface_points = np.concatenate(segments)
    else:
        closest_mesh_points = np.isclose(
            fine_dispersion,
            fermi_level,
            rtol=0.0,
            atol=ENERGY_TOLERANCE,
        )
        if not np.any(closest_mesh_points):
            nearest_index = np.argmin(np.abs(fine_dispersion - fermi_level))
            closest_mesh_points.flat[nearest_index] = True
        fermi_surface_points = np.column_stack(
            (
                context["fine_kx"][closest_mesh_points],
                context["fine_ky"][closest_mesh_points],
            )
        )

    momentum_differences = (
        context["finite_momenta"][np.newaxis, :, :]
        - fermi_surface_points[:, np.newaxis, :]
    )
    momentum_differences = (momentum_differences + np.pi) % (2.0 * np.pi) - np.pi
    distances_squared = np.sum(momentum_differences**2, axis=2)
    distances_to_contour = np.sqrt(np.min(distances_squared, axis=0))
    maximum_contour_distance = 0.99 * np.pi * np.sqrt(2.0) / context["L"]
    return (distances_to_contour <= maximum_contour_distance).reshape(
        context["dispersion"].shape
    )


def _compute_finite_size_gap(context, fermi_level):
    fermi_surface_mask = _find_fermi_surface_mask(context, fermi_level)
    lattice_energy_grid = context["dispersion"]
    neighbor_energy_grids = (
        np.roll(lattice_energy_grid, 1, axis=0),
        np.roll(lattice_energy_grid, -1, axis=0),
        np.roll(lattice_energy_grid, 1, axis=1),
        np.roll(lattice_energy_grid, -1, axis=1),
    )
    neighbor_differences = np.stack(
        [
            np.abs(lattice_energy_grid - neighbor_grid)
            for neighbor_grid in neighbor_energy_grids
        ]
    )
    return float(np.max(neighbor_differences[:, fermi_surface_mask]))


def energy_gap_for_parameters(L, T_PRIME, E_F):
    """Return the maximum four-neighbor energy difference at the Fermi surface."""
    context = _build_gap_context(L, T_PRIME)
    return _compute_finite_size_gap(context, E_F)


def main():
    context = _build_gap_context(L, T_PRIME)
    k = context["k"]
    kx = context["kx"]
    ky = context["ky"]
    dispersion = context["dispersion"]
    fine_k = context["fine_k"]
    fine_dispersion = context["fine_dispersion"]
    finite_energies = context["finite_energies"]
    finite_momenta = context["finite_momenta"]

    fermi_surface_mask = _find_fermi_surface_mask(context, FERMI_LEVEL)
    energy_gap = _compute_finite_size_gap(context, FERMI_LEVEL)
    closest_distance = np.min(np.abs(dispersion - FERMI_LEVEL))

    print(f"L = {L}, T_PRIME = {T_PRIME:g}, Fermi level = {FERMI_LEVEL:g}")
    print(
        "Closest finite-size energy distance to the Fermi level = "
        f"{closest_distance:.12g}"
    )
    print(
        "Finite-size Fermi-surface momenta "
        f"({np.count_nonzero(fermi_surface_mask)}):"
    )
    print(f"Maximum four-neighbor finite-size energy gap = {energy_gap:.12g}")

    dk = 2.0 * np.pi / L
    wrapped_dispersion = np.pad(dispersion, pad_width=1, mode="wrap")
    extent = (
        k[0] - 1.5 * dk,
        k[-1] + 1.5 * dk,
        k[0] - 1.5 * dk,
        k[-1] + 1.5 * dk,
    )
    fig, ax = plt.subplots()
    image = ax.imshow(
        wrapped_dispersion - FERMI_LEVEL,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
        cmap="inferno",
    )
    if np.min(fine_dispersion) < FERMI_LEVEL < np.max(fine_dispersion):
        ax.contour(
            fine_k,
            fine_k,
            fine_dispersion,
            levels=[FERMI_LEVEL],
            colors="white",
            linewidths=1.0,
        )
    ax.scatter(
        kx[fermi_surface_mask],
        ky[fermi_surface_mask],
        marker="o",
        facecolors="none",
        edgecolors="cyan",
        linewidths=1.5,
        s=70,
        label="Finite-size Fermi surface",
    )
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    fig.colorbar(image, ax=ax, label="Energy")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_title(
        rf"$L={L}$, $t'={T_PRIME:g}$, $E_F={FERMI_LEVEL:g}$, "
        rf"$\Delta E={energy_gap:.4g}$"
    )
    fig.tight_layout()

    unique_energies = np.unique(dispersion)
    fermi_level_values = np.linspace(unique_energies[0], unique_energies[-1], 1001)
    transition_levels = 0.5 * (unique_energies[:-1] + unique_energies[1:])
    fermi_level_values = np.unique(
        np.concatenate((fermi_level_values, transition_levels))
    )
    gap_values = np.array(
        [_compute_finite_size_gap(context, level) for level in fermi_level_values]
    )

    fig_gap, ax_gap = plt.subplots()
    ax_gap.plot(fermi_level_values, gap_values)
    ax_gap.set_xlabel(r"Fermi level $E_F$")
    ax_gap.set_ylabel(r"Maximum four-neighbor energy difference $\Delta E$")
    ax_gap.set_title(rf"$L={L}$, $t'={T_PRIME:g}$")
    ax_gap.grid(True, alpha=0.3)
    if unique_energies[0] <= FERMI_LEVEL <= unique_energies[-1]:
        ax_gap.scatter(
            [FERMI_LEVEL],
            [energy_gap],
            color="red",
            zorder=3,
            label=rf"Selected $E_F={FERMI_LEVEL:g}$",
        )
        ax_gap.legend()
    fig_gap.tight_layout()

    lattice_sizes = np.arange(8, 34, 2)
    size_gap_values = [
        energy_gap_for_parameters(lattice_size, T_PRIME, FERMI_LEVEL)
        for lattice_size in lattice_sizes
    ]

    fig_size, ax_size = plt.subplots()
    ax_size.plot(lattice_sizes, size_gap_values, marker="o")

    ax_inv_size = ax_size.twinx()
    ax_inv_size.plot(
        lattice_sizes,
        1.0 / np.array(size_gap_values),
        marker="x",
        c="C1",
    )
    ax_inv_size.set_ylabel(r"$1 / \Delta E$")
    ax_size.set_xticks(lattice_sizes)
    ax_size.set_xlabel(r"System size $L$")
    ax_size.set_ylabel(r"Maximum four-neighbor energy difference $\Delta E$")
    ax_size.set_title(rf"$t'={T_PRIME:g}$, $E_F={FERMI_LEVEL:g}$")
    ax_size.grid(True, alpha=0.3)
    fig_size.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
