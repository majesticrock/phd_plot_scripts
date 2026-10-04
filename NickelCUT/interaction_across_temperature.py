import math
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
# superconductivity
CHANNEL = "superconductivity"

im_show_kwargs = {
    "origin": "lower",
    "aspect": "equal",
    "interpolation": "nearest",
    "cmap": "seismic",
}


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


class TemperatureAnimation:
    def __init__(self):
        self.temperatures = find_available_temperatures(FLOW_PARAMETERS)
        if not self.temperatures:
            raise RuntimeError("No flow data found for the selected parameters.")

        L = FLOW_PARAMETERS["L"]
        N = L * L
        momentum = Momentum(L, 0, L // 2)
        self.matrices = []
        self.final_l_values = []

        for temperature in self.temperatures:
            parameters = {**FLOW_PARAMETERS, "T": temperature}
            data = load_flow_file(**parameters, force_json=False, dense=dense)
            self.matrices.append(
                data["extracted_channels"][-1][CHANNEL][momentum.pos].reshape(L, L) * N
            )
            self.final_l_values.append(data["l_times"][-1])

        vmax = max(np.max(np.abs(matrix)) for matrix in self.matrices)
        if vmax == 0.0:
            vmax = 0.1

        self.fig, ax = plt.subplots()
        self.image = ax.imshow(
            self.matrices[0], vmin=-vmax, vmax=vmax, **im_show_kwargs
        )

        ticks, labels = create_momentum_labels(L)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")

        selected_points = [
            (L // 2, 0),
            (0, L // 2),
            (L // 4, L // 4),
        ]
        point_colors = plt.get_cmap("tab10").colors
        temperature_fig, temperature_ax = plt.subplots()
        for point_index, (x_index, y_index) in enumerate(selected_points):
            color = point_colors[point_index]
            point_label = (
                f"$({labels[x_index][1:-1]}, {labels[y_index][1:-1]})$"
            )
            values = [
                matrix[y_index, x_index]
                for matrix in self.matrices
            ]
            temperature_ax.plot(
                self.temperatures,
                values,
                marker="o",
                color=color,
                label=point_label,
            )
            ax.scatter(
                x_index,
                y_index,
                s=70,
                facecolors="none",
                edgecolors=[color],
                linewidths=1.5,
            )

        temperature_ax.set_xlabel(r"$T$")
        temperature_ax.set_ylabel(r"$V(k_x,k_y)$")
        temperature_ax.set_title(f"{CHANNEL} at each temperature's final flow step")
        temperature_ax.legend(title=r"$(k_x,k_y)$")
        temperature_ax.grid(True, alpha=0.3)
        temperature_fig.tight_layout()

        self.title = ax.set_title("")
        self.update(0)
        self.fig.colorbar(self.image, label=r"$V(k_x,k_y)$")
        self.fig.tight_layout()

        self.paused = False
        self.animation = FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.temperatures),
            interval=400,
            blit=False,
            repeat=True,
        )
        self.fig.canvas.mpl_connect("button_press_event", self.toggle_pause)

    def toggle_pause(self, _event):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused

    def update(self, frame):
        self.image.set_data(self.matrices[frame])
        self.title.set_text(
            f"{CHANNEL}, T={self.temperatures[frame]:g}, "
            f"$\\ell={self.final_l_values[frame]:.3g}$"
        )
        return self.image, self.title


animation = TemperatureAnimation()
plt.show()