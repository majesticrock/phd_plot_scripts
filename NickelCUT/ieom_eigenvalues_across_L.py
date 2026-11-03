import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from load_flow_files import load_flow_file
from nickel_cut_parameters import FLOW_PARAMETERS


SYSTEM_SIZES = (8, 10, 12, 16)
CHANNELS = (
    ("extremal_sc_eigenvalues", "SC", "-"),
    ("extremal_charge_eigenvalues", "Charge", "--"),
    ("extremal_spin_eigenvalues", "Spin", "-."),
)

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(11, 8),
    layout="constrained",
    sharex=True,
)
axes = axes.flat
colors = plt.get_cmap("tab10")
size_handles = [
    Line2D([0], [0], color=colors(index % colors.N), label=f"L={size}")
    for index, size in enumerate(SYSTEM_SIZES)
]

for size_index, system_size in enumerate(SYSTEM_SIZES):
    parameters = {**FLOW_PARAMETERS, "L": system_size}
    data = load_flow_file(
        **parameters,
        resume_num="",
        force_json=False,
        dense=False,
    )
    print(f"Loaded L={system_size} data produced on {data['time']}")

    point_count = data["number_of_data_points"]
    l_times = np.asarray(data["l_times"][:point_count])
    eigenvalues = np.array(
        [data[key][:point_count] for key, _, _ in CHANNELS]
    )
    color = colors(size_index % colors.N)

    for channel_index, (_, label, linestyle) in enumerate(CHANNELS):
        axes[channel_index].plot(
            l_times,
            eigenvalues[channel_index],
            color=color,
        )

    total_eigenvalue = np.sqrt(np.sum(eigenvalues**2, axis=0))
    axes[3].plot(
        l_times,
        total_eigenvalue,
        color=color,
    )

for ax, (_, label, _) in zip(axes, CHANNELS):
    ax.set_title(label)
axes[3].set_title("Total")

for ax in axes:
    ax.set_ylabel(r"$\lambda_\mathrm{ex}$")
    ax.legend(handles=size_handles, title="System size")

for ax in axes[2:]:
    ax.set_xlabel(r"$\ell \cdot t$")

plt.show()
