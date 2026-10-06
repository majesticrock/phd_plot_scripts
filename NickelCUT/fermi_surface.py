import numpy as np
import matplotlib.pyplot as plt
from create_momentum_labels import create_momentum_labels

L=201
T_PRIME=-0.3
E_F=-1.1

def epsilon(kx, ky):
    return -2. * (
    (np.cos(kx) + np.cos(ky)) 
    + 2 * T_PRIME * np.cos(kx) * np.cos(ky)
    )

def f1(kx, ky):
    return np.exp(-4. * (epsilon(kx, ky) - E_F)**2)

k = np.linspace(-np.pi, np.pi, L)
X,Y = np.meshgrid(k, k)
energies = epsilon(X, Y)
max_energy, min_energy = np.max(energies), np.min(energies)
if min_energy - E_F >= 0:
    raise ValueError("Expected min(E) < 0!")
if max_energy - E_F <= 0:
    raise ValueError("Expected max(E) > 0!")

min_segments = 4
max_segments = 10
n_lower_levels = 8
n_upper_levels = 8
lower_fraction = np.linspace(0., 1., n_lower_levels, endpoint=False)[1:]
upper_fraction = np.linspace(0., 1., n_upper_levels, endpoint=False)[1:]
lower_fraction = 1. - (1. - lower_fraction)**1.4
upper_fraction = upper_fraction**2
energy_levels = np.concat([
    min_energy + (E_F - min_energy) * lower_fraction,
    E_F + (max_energy - E_F) * upper_fraction,
])

fig, ax = plt.subplots(layout="constrained", figsize=(9, 8))

#cont = ax.contourf(X, Y, energies, cmap="magma", levels=30)
#cbar = fig.colorbar(cont, ax=ax)
#cbar.set_label(r"$\varepsilon (\mathbf{k})$")
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-np.pi, np.pi])
ax.set_aspect("equal")

twice_k = np.linspace(0., np.pi, L)

def ky_of_kx_arg(E, kx):
    return (-0.5*E - np.cos(kx)) / (1 + 2*T_PRIME*np.cos(kx))
def valid_arg(arg):
    return (-1. <= arg + 1e-12) & (arg <= 1. + 1e-12)

def arc_length(E, begin_x, end_x, steps=200):
    kx = np.linspace(begin_x, end_x, steps)
    args = np.clip(ky_of_kx_arg(E, kx), -1., 1.)
    ky = np.arccos(args)
    
    distances = np.sqrt(np.diff(kx)**2 + np.diff(ky)**2)
    return np.sum(distances)

def constant_energy_line(E):
    arg = ky_of_kx_arg(E, twice_k)
    # arccos is valid in this range
    mask = valid_arg(arg)
    ky = np.empty_like(twice_k)
    ky[~mask] = None
    ky[mask] = np.arccos(np.clip(arg[mask], -1., 1.))
    
    kx = twice_k.copy()
    kx[~mask] = None
    return kx, ky

boundary_points = np.zeros((len(energy_levels), 2, 2))
for i, E in enumerate(energy_levels):
    first_kx = 0.
    first_ky = ky_of_kx_arg(E, first_kx)
    
    if not valid_arg(first_ky):
        last_kx = np.pi
        last_ky = ky_of_kx_arg(E, last_kx)
        
        if not valid_arg(last_ky):
            raise ValueError("Invalid last_ky")
        
        last_ky = np.arccos(np.clip(last_ky, -1., 1.))
        
        first_kx = last_ky
        first_ky = last_kx
    else:
        first_ky = np.arccos(np.clip(first_ky, -1., 1.))
        last_kx = first_ky
        last_ky = first_kx
    
    boundary_points[i, 0] = first_kx, first_ky
    boundary_points[i, 1] = last_kx, last_ky

arc_lengths = np.array([
    arc_length(E, b_point[0,0], b_point[1,0]) 
        for E, b_point in zip(energy_levels, boundary_points)
])

segments_per_level = np.array(np.round([
    min_segments + (arc - arc_lengths.min()) / (arc_lengths.max() - arc_lengths.min()) * (max_segments - min_segments)
        for arc in arc_lengths
]), dtype=int)

saved_points = []

for i in range(len(energy_levels)):
    saved_points.append(np.zeros((segments_per_level[i], 2)))
    
    kx = np.linspace(boundary_points[i,0,0], boundary_points[i,1,0], 200)
    ky = np.arccos(np.clip(ky_of_kx_arg(energy_levels[i], kx), -1., 1.))
    
    distances = np.cumsum(np.sqrt(np.diff(kx)**2 + np.diff(ky)**2))
    distances = np.insert(distances, 0, 0.)
    target_distances = np.linspace(0., arc_lengths[i], segments_per_level[i], endpoint=False)
    
    for t, target in enumerate(target_distances):
        best_idx = np.argmin(np.abs(target - distances))
        saved_points[i][t] = kx[best_idx], ky[best_idx]


for i, points in enumerate(saved_points):
    rotated_points = [points]
    rotated = points
    for _ in range(3):
        rotated = np.column_stack((-rotated[:, 1], rotated[:, 0]))
        rotated_points.append(rotated)
    saved_points[i] = np.concatenate(rotated_points)

for i in range(len(saved_points)):
    for j in range(len(saved_points[i])):
        if np.isclose(saved_points[i][j][0], np.pi):
            saved_points[i][j][0] = -np.pi # momenta are only defined modulo 2 pi
        if np.isclose(saved_points[i][j][1], np.pi):
            saved_points[i][j][1] = -np.pi # momenta are only defined modulo 2 pi
            
        if np.isclose(saved_points[i][j][0], -np.pi):
            saved_points[i][j][0] = -np.pi # snap to -pi
        if np.isclose(saved_points[i][j][1], -np.pi):
            saved_points[i][j][1] = -np.pi # snap to -pi

    saved_points[i] = np.unique(saved_points[i], axis=0)

saved_points.append(np.array([[0., 0.]]))
saved_points.append(np.array([[-np.pi, -np.pi]]))

print("Number of points =", np.sum([len(sp) for sp in saved_points]))

for points in saved_points:
    ax.scatter(*points.T, color="k", zorder=10)

from scipy.spatial import Voronoi, voronoi_plot_2d
flat_points = []
for sp in saved_points:
    flat_points.extend(sp)
flat_points = np.asarray(flat_points)

shifts = np.array([
    [-2*np.pi, -2*np.pi], [-2*np.pi, 0], [-2*np.pi, 2*np.pi],
    [ 0, -2*np.pi], [ 0, 0], [ 0, 2*np.pi],
    [ 2*np.pi, -2*np.pi], [ 2*np.pi, 0], [ 2*np.pi, 2*np.pi],
])
tiled_points = np.concatenate(
    [flat_points + shift for shift in shifts],
    axis=0
)

vor = Voronoi(tiled_points)
voronoi_plot_2d(vor, ax=ax, show_vertices=False)


N = len(flat_points)
areas = np.empty(N)
for i in range(N):
    # Index of the original point in the central tile
    point_idx = 4 * N + i

    # Voronoi region associated with this point
    region_idx = vor.point_region[point_idx]
    region = vor.regions[region_idx]

    # Should be bounded because of the periodic replication
    if -1 in region:
        raise ValueError(f"Unbounded region for point {i}")

    vertices = vor.vertices[region]

    # Shoelace formula
    x = vertices[:, 0]
    y = vertices[:, 1]

    areas[i] = 0.5 * abs(
        np.dot(x, np.roll(y, -1))
        - np.dot(y, np.roll(x, -1))
    )

energy_contours = np.array([
    [*constant_energy_line(EF)] for EF in energy_levels
])
for energy, ec in zip(energy_levels, energy_contours):
    rotated = ec
    if np.isclose(energy, E_F):
        line, = ax.plot(*rotated, color="black")
    else:
        line, = ax.plot(*rotated)
    color = line.get_color()
    for _ in range(3):
        rotated = np.array([-rotated[1], rotated[0]])
        ax.plot(*rotated, color=color)


from matplotlib.collections import PolyCollection

polygons = []
colors = []

for i in range(N):
    idx = 4 * N + i  # point in central tile

    region = vor.regions[vor.point_region[idx]]

    if -1 in region or len(region) == 0:
        continue

    polygon = vor.vertices[region]

    polygons.append(polygon)
    colors.append(areas[i])

collection = PolyCollection(
    polygons,
    array=np.asarray(colors),
    cmap="viridis",
    edgecolor="black",
    linewidth=0.5,
)

#ax.add_collection(collection)

test_contour = ax.contourf(X, Y, f1(X, Y), levels=30, cmap="magma", zorder=0)
cbar = plt.colorbar(test_contour, ax=ax)
cbar.set_label(r"$f_1(\mathbf{k})$")
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)


def voronoi_integrate(func):
    return np.sum(func(flat_points[:, 0], flat_points[:, 1]) * areas)

def coarse_integrate(func, L):
    k = np.linspace(-np.pi, np.pi, L, endpoint=False)
    X, Y = np.meshgrid(k, k)
    dk = k[1] - k[0]
    return np.sum(func(X, Y)) * dk**2

import scipy.integrate as sint
def test(func):
    correct = sint.dblquad(func, -np.pi, np.pi, -np.pi, np.pi)[0]
    print("Scipy:", sint.dblquad(func, -np.pi, np.pi, -np.pi, np.pi))
    
    def _run(routine, name):
        val = routine(func)
        print(f"{name}:", val, "   error =", np.abs(correct - val))
        
    _run(voronoi_integrate, "Voronoi")
    _run(lambda p: coarse_integrate(p, 10), "Coarse 10")
    _run(lambda p: coarse_integrate(p, 20), "Coarse 20")

test(f1)

plt.show()