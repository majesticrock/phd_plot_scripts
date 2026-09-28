import numpy as np
import matplotlib.pyplot as plt

from Momentum import Gamma, Momentum
from load_full_flow_file import load_full_flow_state
from nickel_cut_parameters import FLOW_PARAMETERS


data = load_full_flow_state(**FLOW_PARAMETERS, resume_num="", force_json=False)
L = data["L"]
N = L * L


def compute_occupation_numbers(beta):
	"""Compute occupations from the stored dispersion at inverse temperature beta."""
	dispersion = np.asarray(data["dispersion"])
	if beta >= 0.0:
		return 1.0 / (1.0 + np.exp(beta * dispersion))

	occupation_numbers = np.zeros_like(dispersion, dtype=float)
	zero_energy = np.isclose(dispersion, 0.0, atol=1e-10, rtol=0.0)
	occupation_numbers[dispersion < 0.0] = 1.0
	occupation_numbers[zero_energy] = 0.5
	return occupation_numbers


def compute_dynamical_matrix(flow_state, occupation_numbers, x):
	"""Construct the C++ IEOM dynamical matrix for transfer momentum x."""
	dynamical_matrix = np.zeros((N, N), dtype=float)
	differing_spin = flow_state["interactions_differing_spin"]
	same_spin = flow_state["interactions_same_spin"]
	epsilon_tilde = flow_state["epsilon_tilde"]
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
						- same_spin[k.pos, K.pos, (K - k - x).pos]
						+ same_spin[k.pos, K.pos, gamma.pos]
					) * occupation_numbers[K.pos]

			row_factor = 1.0 - occupation_numbers[k.pos] - occupation_numbers[kx]
			dynamical_matrix[k.pos, k.pos] = 2.0 * row_factor * diagonal

			for l_y in range(L):
				for l_x in range(L):
					l = Momentum(L, l_x, l_y)
					col_factor = (
						1.0 - occupation_numbers[l.pos]
						- occupation_numbers[(l + x).pos]
					)
					dynamical_matrix[k.pos, l.pos] += (
						2.0
						* differing_spin[(-k - x).pos, k.pos, (k - l).pos]
						* row_factor
						* col_factor
					)

	return dynamical_matrix


def compute_norm_matrix(occupation_numbers, x):
	"""Construct the diagonal IEOM norm matrix for transfer momentum x."""
	norm_diagonal = np.empty(N, dtype=float)

	for y in range(L):
		for x_index in range(L):
			k = Momentum(L, x_index, y)
			norm_diagonal[k.pos] = (
				occupation_numbers[k.pos]
				+ occupation_numbers[(k + x).pos]
				- 1.0
			)

	return np.diag(norm_diagonal)

def chi_ieom(flow_state, occupation_numbers, x):
	N_mat = compute_norm_matrix(occupation_numbers, x)
	M_mat = compute_dynamical_matrix(flow_state, occupation_numbers, x)
	return np.sum(N_mat @ np.linalg.inv(M_mat) @ N_mat) / N


def plot_divergence_eigenvector(result, x, ax=None):
	"""Plot a divergence eigenvector on the two-dimensional momentum grid."""
	if result is None:
		return None

	if ax is None:
		fig, ax = plt.subplots()
	else:
		fig = ax.get_figure()

	eigenvector = np.asarray(result["eigenvector"]).reshape(L, L)
	color_limit = np.max(np.abs(eigenvector))
	if color_limit == 0.0:
		color_limit = 1.0
	image = ax.imshow(
		eigenvector,
		origin="lower",
		interpolation="nearest",
		cmap="seismic",
		vmin=-color_limit,
		vmax=color_limit,
	)
	ax.set_xlabel(r"$k_x$")
	ax.set_ylabel(r"$k_y$")
	ax.set_title(
		f"IEOM mode at x=({x.x}, {x.y}), T={result['temperature']:.6g}"
	)
	fig.colorbar(image, ax=ax, label=r"$v(\mathbf{k})$")
	fig.tight_layout()
	return fig, ax


def find_ieom_divergence(
	flow_state,
	x,
	beta_min=0.05,
	beta_max=20.0,
	beta_step=0.1,
	beta_tolerance=1e-5,
	weight_tolerance=1e-8
):
	"""Find the first active zero crossing of an IEOM eigenvalue as beta rises."""
	if beta_min <= 0.0 or beta_max <= beta_min or beta_step <= 0.0:
		raise ValueError("Require 0 < beta_min < beta_max and beta_step > 0")
	print(
		f"Searching for an IEOM divergence at x=({x.x}, {x.y}) "
		f"from beta={beta_min:g} to {beta_max:g}"
	)

	def diagonalize(beta):
		occupations = compute_occupation_numbers(beta)
		norm_diagonal = np.diag(compute_norm_matrix(occupations, x))
		dynamical_matrix = compute_dynamical_matrix(flow_state, occupations, x)
		eigenvalues, eigenvectors = np.linalg.eigh(dynamical_matrix)

		
		min_weight = np.sum((norm_diagonal * eigenvectors[:,0])**2) / N
		i=0
		while np.abs(min_weight) < weight_tolerance:
			i+=1
			min_weight = np.sum((norm_diagonal * eigenvectors[:,i])**2) / N
		
		min_value = eigenvalues[i]
		min_vector = np.linalg.pinv(norm_diagonal) * eigenvectors[:,i]
    
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



#beta = 8.

beta, f, vec, weight = find_ieom_divergence(data, Momentum(L, L//2, L//2),
					 beta_min=1.,
					 beta_max=20.,
					 beta_tolerance=0.01,
					 weight_tolerance=1e-11)

fig_vector, ax_vector = plt.subplots(layout="constrained")
ax_vector.set_title(f"$\\beta={beta}$")
im = ax_vector.imshow(vec.reshape(L, L), aspect="equal")
cbar = fig_vector.colorbar(im, ax=ax_vector)
cbar.set_label("Eigenvector")
ax_vector.set_xlabel("$q_x$")
ax_vector.set_xlabel("$q_y$")

beta *= 1.1
occupation_numbers = compute_occupation_numbers(beta)

n = L // 2
path = (
	[Momentum(L, n, y) for y in range(n, -1, -1)]
	+ [Momentum(L, x, 0) for x in range(n - 1, -1, -1)]
	+ [Momentum(L, index, index) for index in range(1, n + 1)]
)
chi_values = np.asarray([
	chi_ieom(data, occupation_numbers, momentum) for momentum in path
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
