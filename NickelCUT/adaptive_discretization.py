import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

coarse_L = 5
largest_cell_units = 3
fine_band_inward_extension = 1

def distance_to_diagonals(x, y, unit):
	"""Distance to the four diagonal segments across the square."""
	endpoints = (
		((np.pi, 0), (0, np.pi)),
		((np.pi, 0), (0, -np.pi)),
		((-np.pi, 0), (0, np.pi)),
		((-np.pi, 0), (0, -np.pi)),
	)
	distances = []
	point = np.array([x, y])
	for start, end in endpoints:
		start = np.asarray(start)
		segment = np.asarray(end) - start
		fraction = np.clip(np.dot(point - start, segment) / np.dot(segment, segment), 0, 1)
		nearest_point = start + fraction * segment
		offset = point - nearest_point
		distance = np.linalg.norm(offset)
		if np.dot(offset, -nearest_point) > 0:
			distance = max(0, distance - fine_band_inward_extension * unit)
		distances.append(distance)
	return min(distances)


def target_spacing(x, y, unit):
	distance = distance_to_diagonals(x, y, unit)
	maximum_distance = np.pi / np.sqrt(2)
	t = np.clip(2 * distance / maximum_distance, 0, 1)
	smooth_transition = t**3 * (10 + t * (-15 + 6 * t))
	return unit * (1 + (largest_cell_units - 1) * smooth_transition)


def adaptive_grid():
	lattice_size = coarse_L * largest_cell_units
	lattice_size += lattice_size % 2
	unit = 2 * np.pi / lattice_size
	finest_grid = np.linspace(-np.pi, np.pi, 2 * lattice_size, endpoint=False)
	occupied = np.zeros((lattice_size, lattice_size), dtype=bool)
	cells = []
	cell_indices = []

	def cell_center(i, j, size):
		return finest_grid[(2 * i + size - 1) % (2 * lattice_size)], finest_grid[(2 * j + size - 1) % (2 * lattice_size)]

	def physical_cell(i, j, size):
		x0 = -np.pi - unit / 2 + i * unit
		y0 = -np.pi - unit / 2 + j * unit
		return x0, x0 + size * unit, y0, y0 + size * unit

	def symmetric_orbit(i, j, size):
		orbit = set()
		for swapped in (False, True):
			x, y = (j, i) if swapped else (i, j)
			for reflect_x in (False, True):
				for reflect_y in (False, True):
					reflected_x = (1 - size - x) % lattice_size if reflect_x else x
					reflected_y = (1 - size - y) % lattice_size if reflect_y else y
					orbit.add((reflected_x, reflected_y))
		return orbit

	def block_indices(i, j, size):
		return {
			(x % lattice_size, y % lattice_size)
			for x in range(i, i + size)
			for y in range(j, j + size)
		}

	def add_symmetric_blocks(i, j, size):
		orbit = symmetric_orbit(i, j, size)
		claimed = set()
		for block_i, block_j in orbit:
			center = cell_center(block_i, block_j, size)
			if target_spacing(*center, unit) < size * unit * (1 - 1e-12):
				return False
			indices = block_indices(block_i, block_j, size)
			if claimed.intersection(indices):
				return False
			if any(occupied[x, y] for x, y in indices):
				return False
			claimed.update(indices)

		for block_i, block_j in orbit:
			for x, y in block_indices(block_i, block_j, size):
				occupied[x, y] = True
			cells.append(physical_cell(block_i, block_j, size))
			cell_indices.append((block_i, block_j, size))
		return True

	# Align odd-sized coarse blocks with the reflection-symmetric phase.
	coarse_offset = (largest_cell_units + 1) // 2 if largest_cell_units % 2 else 0
	for i in range(coarse_offset, lattice_size + coarse_offset, largest_cell_units):
		i %= lattice_size
		for j in range(coarse_offset, lattice_size + coarse_offset, largest_cell_units):
			add_symmetric_blocks(i, j % lattice_size, largest_cell_units)

	medium_candidates = []
	for i in range(lattice_size):
		for j in range(lattice_size):
			center = cell_center(i, j, 2)
			spacing = target_spacing(*center, unit)
			if spacing >= 2 * unit * (1 - 1e-12):
				medium_candidates.append((-spacing, i, j))

	for _, i, j in sorted(medium_candidates):
		add_symmetric_blocks(i, j, 2)

	for i in range(lattice_size):
		for j in range(lattice_size):
			if not occupied[i, j]:
				add_symmetric_blocks(i, j, 1)

	points = np.asarray([
		(finest_grid[(2 * i + size - 1) % (2 * lattice_size)], finest_grid[(2 * j + size - 1) % (2 * lattice_size)])
		for i, j, size in cell_indices
	])
	return points, cells, unit, finest_grid


grid, cells, unit, finest_grid = adaptive_grid()

segments = set()
for x0, x1, y0, y1 in cells:
	for x_shift in (-2 * np.pi, 0, 2 * np.pi):
		for y_shift in (-2 * np.pi, 0, 2 * np.pi):
			shifted_x0, shifted_x1 = x0 + x_shift, x1 + x_shift
			shifted_y0, shifted_y1 = y0 + y_shift, y1 + y_shift
			if shifted_x1 < -np.pi or shifted_x0 > np.pi or shifted_y1 < -np.pi or shifted_y0 > np.pi:
				continue
			segments.update((
				tuple(sorted(((shifted_x0, shifted_y0), (shifted_x1, shifted_y0)))),
				tuple(sorted(((shifted_x0, shifted_y1), (shifted_x1, shifted_y1)))),
				tuple(sorted(((shifted_x0, shifted_y0), (shifted_x0, shifted_y1)))),
				tuple(sorted(((shifted_x1, shifted_y0), (shifted_x1, shifted_y1)))),
			))

L = 201
T_PRIME = -0.3


def epsilon(kx, ky):
	return -2 * ((np.cos(kx) + np.cos(ky)) + 2 * T_PRIME * np.cos(kx) * np.cos(ky))


k = np.linspace(-np.pi, np.pi, L)
X, Y = np.meshgrid(k, k)
energies = epsilon(X, Y)

fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
contour = ax.contourf(X, Y, energies, cmap="viridis", levels=30)
colorbar = fig.colorbar(contour, ax=ax)
colorbar.set_label(r"$\varepsilon (\mathbf{k})$")
ax.add_collection(LineCollection(sorted(segments), colors="black", linewidths=0.55))
ax.scatter(grid[:, 0], grid[:, 1], s=50, color="tab:red", edgecolors="white", linewidths=0.25, zorder=2)
print("Number of points:", len(grid))
ax.set(xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi), aspect="equal")
ax.set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
ax.set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
ax.set_xlabel("$k_x$")
ax.set_ylabel("$k_y$")

plt.show()