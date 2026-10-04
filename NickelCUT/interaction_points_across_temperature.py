import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mrock.get_data import DataLoader

from create_momentum_labels import create_momentum_labels
from load_flow_files import load_flow_file
from nickel_cut_parameters import FLOW_PARAMETERS
from Momentum import Momentum

dense = True
FLOW_FILE_PATTERN = f"{'dense_' if dense else ''}flow.json.gz*"
FLOW_FILE_RE = re.compile(fr"^{FLOW_FILE_PATTERN}\d*$")

# density_wave_differing | density_wave_same
# single_particle_energy_differing | single_particle_energy_same
# superconductivity | AFM
CHANNELS = ["superconductivity"]#, "single_particle_energy_differing", "AFM")


def find_available_temperatures(parameters):
    loader = DataLoader()
    base_directory = loader._to_path(
        "nickel_cut",
        parameters["subdir"],
        L=parameters["L"],
    )
    relative_base = base_directory.relative_to(loader.data_dir)
    flow_files = loader.get_all_files(relative_base, FLOW_FILE_PATTERN)

    fixed_parameters = {
        key: parameters[key] for key in ("L", "U_0", "tprime", "E_F")
    }
    available = set()

    for flow_file in flow_files:
        path = Path(flow_file)
        if not FLOW_FILE_RE.match(path.name):
            continue

        components = {}
        for part in path.parent.parts:
            if "=" in part:
                key, value = part.split("=", 1)
                components[key] = value

        try:
            matches_fixed_parameters = all(
                math.isclose(
                    float(components[key]),
                    float(value),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for key, value in fixed_parameters.items()
            )
            temperature = float(components["T"])
        except (KeyError, ValueError):
            continue

        if matches_fixed_parameters:
            available.add(temperature)

    return sorted(available)


temperatures = find_available_temperatures(FLOW_PARAMETERS)
if not temperatures:
    raise RuntimeError("No flow data found for the selected parameters.")

L = FLOW_PARAMETERS["L"]
N = L * L
momentum = Momentum(L, 0, L // 2)
selected_points = [
    (L // 2, 0),
    (0, L // 2),
    (L // 4, L // 4),
]
point_values = {
    channel: {point: [] for point in selected_points}
    for channel in CHANNELS
}

for temperature in temperatures:
    parameters = {**FLOW_PARAMETERS, "T": temperature}
    data = load_flow_file(**parameters, force_json=False, dense=dense)
    final_channels = data["extracted_channels"][-1]
    for channel in CHANNELS:
        if channel != "AFM":
            final_matrix = (
                final_channels[channel][momentum.pos].reshape(L, L) * N
            )
        else:
            final_matrix = (
                final_channels["density_wave_differing"][momentum.pos].reshape(L, L)
                - 2
                * final_channels["density_wave_same"][momentum.pos].reshape(L, L)
            ) * N
        for x_index, y_index in selected_points:
            point_values[channel][x_index, y_index].append(
                final_matrix[y_index, x_index]
            )

_, momentum_labels = create_momentum_labels(L)
line_styles = ("-", "--", ":")
temperature_markers = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p")
fig, ax = plt.subplots(figsize=(10, 6))
line_handles = []

for c, channel in enumerate(CHANNELS):
    for point_index, (x_index, y_index) in enumerate(selected_points):
        values = point_values[channel][x_index, y_index]
        line_style = line_styles[point_index % len(line_styles)]
        color = f"C{c}"
        ax.plot(
            temperatures,
            values,
            color=color,
            linestyle=line_style,
            linewidth=1.5,
        )
        for temperature_index, (temperature, value) in enumerate(
            zip(temperatures, values)
        ):
            ax.scatter(
                temperature,
                value,
                color=color,
                marker=temperature_markers[
                    temperature_index % len(temperature_markers)
                ],
                s=64,
                zorder=3,
            )

        point_label = (
            f"$({momentum_labels[x_index][1:-1]}, "
            f"{momentum_labels[y_index][1:-1]})$"
        )
        line_handles.append(
            Line2D(
                [],
                [],
                color=color,
                linestyle=line_style,
                label=f"{channel} {point_label}",
            )
        )

ax.set_xlabel(r"$T$")
ax.set_ylabel(r"$V(k_x,k_y)$")
ax.set_title("Final-flow interaction versus temperature")
ax.grid(True, alpha=0.3)
fig.subplots_adjust(right=0.68)

series_legend = ax.legend(
    handles=line_handles,
    title="Channel / momentum",
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
)
ax.add_artist(series_legend)
temperature_handles = [
    Line2D(
        [],
        [],
        color="0.3",
        marker=temperature_markers[index % len(temperature_markers)],
        linestyle="None",
        label=f"T={temperature:g}",
    )
    for index, temperature in enumerate(temperatures)
]
ax.legend(
    handles=temperature_handles,
    title="Temperature",
    loc="lower left",
    bbox_to_anchor=(1.02, 0),
)

plt.show()