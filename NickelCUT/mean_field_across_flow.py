import gzip
import json

import matplotlib.pyplot as plt
import numpy as np

from mrock.get_data import DataLoader, nickel_cut_params
from nickel_cut_parameters import FLOW_PARAMETERS
from create_momentum_labels import create_momentum_labels


MEAN_FIELD_FILE_NAME = "mean_field_across_flow.json.gz"
PLOT_L_INDICES = [0, 5, 10, 13, -1]

PLOTS = [
	("Delta_AFM", r"$\Delta_{\mathrm{AFM}}(\mathbf{k})$"),
	("Delta_CDW", r"$\Delta_{\mathrm{CDW}}(\mathbf{k})$"),
	("Delta_SC", r"$\Delta_{\mathrm{SC}}(\mathbf{k})$"),
]


def load_mean_field_across_flow():
	parameters = {
		key: value
		for key, value in FLOW_PARAMETERS.items()
		if key != "subdir"
	}
	file_path = DataLoader()._to_path(
		"nickel_cut",
		FLOW_PARAMETERS["subdir"],
		**nickel_cut_params(**parameters),
	) / MEAN_FIELD_FILE_NAME

	with gzip.open(file_path, "rt") as compressed_file:
		return json.load(compressed_file)


def plot_order_parameter(ax, values, title, value_limit, lattice_size, ticks, labels):
	image = ax.imshow(
		np.asarray(values).reshape(lattice_size, lattice_size),
		origin="lower",
		extent=(-np.pi, np.pi, -np.pi, np.pi),
		interpolation="nearest",
		cmap="seismic",
		vmin=-value_limit,
		vmax=value_limit,
	)
	ax.set_title(title)
	ax.set_xlabel(r"$k_x$")
	ax.set_ylabel(r"$k_y$")
	ax.set_xticks(ticks, labels=labels)
	ax.set_yticks(ticks, labels=labels)
	return image


data = load_mean_field_across_flow()
solutions = data["solutions"]
lattice_size = data.get("L", FLOW_PARAMETERS["L"])

indices_to_plot = list(dict.fromkeys([*PLOT_L_INDICES]))
momentum_ticks, momentum_labels = create_momentum_labels(lattice_size)
momentum_ticks = np.pi * (2 * momentum_ticks / lattice_size - 1)

value_limit = max(
	0.1,
	max(
		np.max(np.abs(np.asarray(solutions[index][key])))
		for index in indices_to_plot
		for key, _ in PLOTS
	),
)

fig, axes = plt.subplots(
	len(indices_to_plot),
	len(PLOTS),
	figsize=(10, 12),
	squeeze=False,
	layout="constrained",
)

for row, index in enumerate(indices_to_plot):
	solution = solutions[index]
	flow_time = solution.get("l", data["l_times"][index])
	for column, (key, title) in enumerate(PLOTS):
		row_title = rf"$i={index},\ \ell={flow_time:g}$  " if column == 0 else ""
		image = plot_order_parameter(
			axes[row, column],
			solution[key],
			row_title + title,
			value_limit,
			lattice_size,
			momentum_ticks,
			momentum_labels,
		)
	fig.colorbar(image, ax=axes[row, :], label=r"$\Delta(\mathbf{k})$")

plt.show()
