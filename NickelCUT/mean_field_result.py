import matplotlib.pyplot as plt
import numpy as np

from mrock.get_data import DataLoader, nickel_cut_params
from nickel_cut_parameters import FLOW_PARAMETERS
from create_momentum_labels import create_momentum_labels


MEAN_FIELD_FILE_NAME = "mean_field_solution.json.gz"


def load_mean_field_solution(subdir=""):
	return DataLoader().load_panda(
		"nickel_cut",
		subdir,
		MEAN_FIELD_FILE_NAME,
		print_date=False,
		**nickel_cut_params(**FLOW_PARAMETERS),
	)


def plot_order_parameter(ax, values, title, value_limit, cmap="seismic"):
	values = np.asarray(values)
	image = ax.imshow(
		values.reshape(L, L),
		origin="lower",
		extent=(-np.pi, np.pi, -np.pi, np.pi),
		interpolation="nearest",
		cmap=cmap,
		vmin=-value_limit,
		vmax=value_limit,
	)
	ax.set_title(title)
	ax.set_xlabel(r"$k_x$")
	ax.set_ylabel(r"$k_y$")
	ax.set_xticks(momentum_ticks, labels=momentum_labels)
	ax.set_yticks(momentum_ticks, labels=momentum_labels)
	return image


data = load_mean_field_solution()
L = FLOW_PARAMETERS["L"]
momentum_ticks, momentum_labels = create_momentum_labels(L)
momentum_ticks = np.pi * (2 * momentum_ticks / L - 1)

plots = [
	("Delta_AFM", r"$\Delta_{\mathrm{AFM}}(\mathbf{k})$"),
	("Delta_CDW", r"$\Delta_{\mathrm{CDW}}(\mathbf{k})$"),
	("Delta_SC", r"$\Delta_{\mathrm{SC}}(\mathbf{k})$"),
]

value_limit = max(
	0.1,
	max(np.max(np.abs(np.asarray(data[key]))) for key, _ in plots),
)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
for ax, (key, title) in zip(axes.flat, plots):
	image = plot_order_parameter(ax, data[key], title, value_limit)
fig.colorbar(image, ax=axes, label=r"$\Delta(\mathbf{k})$")

plt.show()
