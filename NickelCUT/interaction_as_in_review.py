import numpy as np
import matplotlib.pyplot as plt
from load_full_flow_file import load_full_flow_state
from nickel_cut_parameters import FLOW_PARAMETERS

from Momentum import Momentum


def extract_fermi_surface(dispersion):
    """Extract zero-dispersion momenta in counter-clockwise order.

    The returned momenta are the grid points nearest to the interpolated
    zero contour.  The contour is treated as periodic in both directions.
    """
    dispersion = np.asarray(dispersion)
    L = int(np.sqrt(dispersion.size))
    if dispersion.ndim != 1 or L * L != dispersion.size:
        raise ValueError("dispersion must be a flat array of length L**2")
    if not np.all(np.isfinite(dispersion)):
        raise ValueError("dispersion must contain only finite values")

    dispersion_2d = dispersion.reshape(L, L)
    tiled_dispersion = np.tile(dispersion_2d, (3, 3))
    grid = np.arange(-L, 2 * L)
    contour = plt.contour(grid, grid, tiled_dispersion, levels=[0])
    paths = contour.allsegs[0]
    plt.close(contour.axes.figure)
    if not paths:
        raise ValueError("dispersion has no zero contour")

    points = []
    seen = set()
    zero_tolerance = 1e-12 * max(1.0, np.max(np.abs(dispersion)))
    zero_positions = np.flatnonzero(np.abs(dispersion) <= zero_tolerance)
    zero_positions = set(zero_positions.tolist())
    for position in zero_positions:
        point = Momentum(L, position % L, position // L)
        points.append(point)
        seen.add(point.pos)

    base_path = min(paths, key=lambda path: np.linalg.norm(path.mean(axis=0)))
    boundary_paths = [
        path
        for path in paths
        if path is not base_path
        and np.any(
            np.isclose(path[:, 0], 0)
            | np.isclose(path[:, 0], L)
            | np.isclose(path[:, 1], 0)
            | np.isclose(path[:, 1], L)
        )
    ]
    paths_to_process = [(base_path, False)]
    if boundary_paths:
        neighboring_path = min(
            boundary_paths,
            key=lambda path: np.linalg.norm(path.mean(axis=0) - np.array([L, 0])),
        )
        paths_to_process.append((neighboring_path, True))
    for path, boundary_only in paths_to_process:
        path_points = []
        path_seen = set()
        for x, y in path:
            if boundary_only and not (
                np.isclose(x, 0)
                or np.isclose(x, L)
                or np.isclose(y, 0)
                or np.isclose(y, L)
            ):
                continue
            x_candidates = np.floor(x).astype(int) + np.array([0, 1])
            y_candidates = np.floor(y).astype(int) + np.array([0, 1])
            candidates = [
                (candidate_x % L, candidate_y % L)
                for candidate_x in x_candidates
                for candidate_y in y_candidates
            ]
            x_index, y_index = min(
                candidates,
                key=lambda candidate: abs(dispersion[candidate[0] + L * candidate[1]]),
            )
            point = Momentum(L, x_index, y_index)
            if point.pos not in path_seen:
                path_points.append(point)
                path_seen.add(point.pos)

        for point in path_points:
            if point.pos in zero_positions:
                if point.pos not in seen:
                    points.append(point)
                    seen.add(point.pos)
            else:
                points.append(point)

    if not points:
        raise ValueError("could not extract momenta from the zero contour")

    target = np.array([-np.pi, 0])
    fs_angles = np.array([
        np.arctan2(point.ky, point.kx)
        for point in points
    ])
    order = np.argsort((fs_angles - np.pi) % (2 * np.pi))
    points = np.array(points)[order]
    start = np.argmin([
        (point.kx - target[0]) ** 2 + (point.ky - target[1]) ** 2
        for point in points
    ])
    return np.roll(points, -start, axis=0)


def complete_fermi_surface_quarters(points):
    """Complete the discrete grid representation in each FS quarter."""
    points = list(points)
    L = points[0].L
    center = L // 2

    has_seam_duplicate = any(
        first.pos == second.pos
        for first, second in zip(points, points[1:])
    )
    if not has_seam_duplicate:
        return np.array(points)

    if len(points) > 1 and points[0].pos == points[1].pos:
        points[1] = Momentum(L, points[1].x + 1, points[1].y)

    duplicate_axis = next(
        (
            index
            for index in range(len(points) - 1)
            if points[index].x == center
            and points[index].y == 0
            and points[index].pos == points[index + 1].pos
        ),
        None,
    )
    if duplicate_axis is not None:
        point = points[duplicate_axis]
        shifted_point = Momentum(L, point.x, point.y + 1)
        points[duplicate_axis] = shifted_point

    index = 0
    while index < len(points) - 1:
        first = points[index]
        second = points[index + 1]
        if first.x == second.x and first.y < center < second.y:
            points.insert(
                index + 1,
                Momentum(L, first.x + (1 if first.x > center else -1), center),
            )
            index += 1
        elif (
            first.y == second.y
            and first.x > center > second.x
            and first.y > center
        ):
            points.insert(index + 1, Momentum(L, center, first.y + 1))
            index += 1
        index += 1

    return np.array(points)


data = load_full_flow_state(subdir="", **FLOW_PARAMETERS, resume_num="", force_json=False)

L = data["L"]

FS_points = complete_fermi_surface_quarters(
    extract_fermi_surface(data["dispersion"])
)
FS_indices = np.array([point.pos for point in FS_points])
fix_k3 = Momentum(L, 0, L//2)
Q_indices = np.array([ (point - fix_k3).pos for point in FS_points ])

im_show_kwargs = {
    "origin":         "lower",
    "aspect":         "equal",
    "interpolation" : "nearest",
    "cmap" :          "seismic"
}

fig, (ax_fs, ax) = plt.subplots(ncols=2, figsize=(12, 5), layout="constrained")

perm = np.array([
    ((-x) % L) + L*((-y) % L)
    for y in range(L)
    for x in range(L)
])

V = data["interactions_differing_spin"][FS_indices[None,:], FS_indices[:,None], Q_indices[:,None]] * L*L
vmax = np.max(np.abs(V))
if vmax == 0.0:
    vmax += 0.1
im = ax.imshow(V, vmin=-vmax, vmax=vmax, **im_show_kwargs)

ax.set_xlabel(r"$p_i$")
ax.set_ylabel(r"$k_i$")
fig.colorbar(im, ax=ax, label=r"$U(\mathbf{k},\mathbf{p},\mathbf{q})$")

dispersion_2d = data["dispersion"].reshape(L, L)
dispersion_vmax = np.max(np.abs(dispersion_2d))
if dispersion_vmax == 0.0:
    dispersion_vmax = 0.1
half_grid_step = np.pi / L
im_fs = ax_fs.imshow(
    dispersion_2d,
    extent=(
        -np.pi - half_grid_step,
        np.pi - half_grid_step,
        -np.pi - half_grid_step,
        np.pi - half_grid_step,
    ),
    vmin=-dispersion_vmax,
    vmax=dispersion_vmax,
    **im_show_kwargs,
)
fig.colorbar(im_fs, ax=ax_fs, label=r"$\varepsilon(\mathbf{k})$")
ax_fs.scatter(
    [point.kx for point in FS_points],
    [point.ky for point in FS_points],
    color="black",
    zorder=2,
)
for number, point in enumerate(FS_points):
    ax_fs.annotate(
        str(number),
        (point.kx, point.ky),
        xytext=(4, 4),
        textcoords="offset points",
    )
ax_fs.set_xlabel(r"$k_x$")
ax_fs.set_ylabel(r"$k_y$")
ax_fs.set_aspect("equal")

plt.show()