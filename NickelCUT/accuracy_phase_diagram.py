import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import numpy as np

from finite_size_energy_gap import energy_gap_for_parameters
from load_flow_files import load_flow_file
from mrock.get_data import DataLoader
from nickel_cut_parameters import FLOW_PARAMETERS


SYSTEM_SIZES = (10, 12, 14, 16)
FLOW_FILE_RE = re.compile(r"^flow\.json\.gz$")
CHANNELS = (
    ("extremal_sc_eigenvalues", "SC", "tab:red", "o"),
    ("extremal_charge_eigenvalues", "Charge", "tab:blue", "s"),
    ("extremal_spin_eigenvalues", "Spin", "tab:green", "v"),
)
METZNER_TC = np.array([
    [-0.7450836820083683, 0.968416, 0.1517948717948718],
    [-0.7989539748953975, 0.946608, 0.14991452991452991],
    [-0.8308577405857741, 0.933240, 0.14991452991452991],
    [-0.8998953974895398, 0.902971, 0.11504273504273504],
    [-0.9506276150627615, 0.879326, 0.11452991452991454],
    [-1.0003138075313807, 0.854784, 0.10974358974358975],
    [-1.1498953974895398, 0.767673, 0.06444444444444444],
    [-1.2006276150627615, 0.725472, 0.039487179487179475],
    [-1.2524058577405857, 0.682998, 0.024444444444444435],
    [-1.297907949790795, 0.652862, 0.014871794871794852],
    [-1.349163179916318, 0.621974, 0.010256410256410248],
]).T
METZNER_DWAVE = np.array([
    [-1.121018062397373, 0.786820, 0.07020134228187919],
    [-1.0996715927750411, 0.800015, 0.07221476510067114],
    [-1.05, 0.828544, 0.04751677852348993],
    [-1.000328407224959, 0.854771, 0.04228187919463086],
    [-0.9506568144499179, 0.879313, 0.0006711409395973034],
]).T


def _available_points():
    loader = DataLoader()
    points_by_size = {}

    for system_size in SYSTEM_SIZES:
        base_directory = loader._to_path(
            "nickel_cut",
            FLOW_PARAMETERS["subdir"],
            L=system_size,
        )
        relative_base = base_directory.relative_to(loader.data_dir)
        flow_files = loader.get_all_files(relative_base, "flow.json.gz")

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
                        float(FLOW_PARAMETERS[key]),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    for key in ("U_0", "tprime")
                )
                temperature = float(components["T"])
                filling = float(components["filling"])
            except (KeyError, ValueError):
                continue

            if matches_fixed_parameters:
                points_by_size.setdefault((temperature, filling), set()).add(
                    system_size
                )

    return points_by_size


def _channel_values_at_threshold(system_size, temperature, filling):
    parameters = {
        **FLOW_PARAMETERS,
        "L": system_size,
        "T": temperature,
        "filling": filling,
    }
    data = load_flow_file(**parameters, resume_num="", force_json=False, dense=False)
    point_count = int(data["number_of_data_points"])
    l_times = np.asarray(data["l_times"][:point_count], dtype=float)
    if point_count == 0 or l_times.size == 0:
        raise ValueError(
            f"No flow points in L={system_size}, T={temperature:g}, "
            f"filling={filling:g}."
        )

    energy_gap = energy_gap_for_parameters(
        system_size,
        FLOW_PARAMETERS["tprime"],
        data["E_F"],
    )
    if not np.isfinite(energy_gap) or energy_gap <= 0.0:
        raise ValueError(
            f"Invalid finite-size energy gap {energy_gap!r} in "
            f"L={system_size}, T={temperature:g}, filling={filling:g}."
        )

    threshold_index = int(np.argmin(np.abs(l_times - 1.0 / energy_gap)))
    values = np.array([
        np.asarray(data[key][:point_count], dtype=float)[threshold_index]
        for key, _, _, _ in CHANNELS
    ])
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Non-finite channel eigenvalue in L={system_size}, "
            f"T={temperature:g}, filling={filling:g}."
        )
    return values


def _opacity(mean, std):
    if mean <= 0.0 or not np.isfinite(std):
        return 1.0
    if std <= 0.0:
        return 0.0
    standard_deviations_to_zero = mean / std
    return 1.0 / (1.0 + standard_deviations_to_zero)


points_by_size = _available_points()
if not points_by_size:
    raise RuntimeError(
        "No flow data found for the configured parameters and system sizes."
    )

points = []
for (temperature, filling), available_sizes in sorted(points_by_size.items()):
    values_by_size = [
        _channel_values_at_threshold(size, temperature, filling)
        for size in SYSTEM_SIZES
        if size in available_sizes
    ]
    values_by_size = np.asarray(values_by_size)
    means = np.mean(values_by_size, axis=0)
    if len(values_by_size) > 1:
        stds = np.std(values_by_size, axis=0, ddof=1)
    else:
        stds = np.full(len(CHANNELS), np.nan)
        print(
            f"Only one system size has data at T={temperature:g}, "
            f"filling={filling:g}; showing full-opacity markers."
        )
    points.append((temperature, filling, means, stds))

fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
marker_offsets = np.linspace(-0.006, 0.006, len(CHANNELS))
channel_handles = []

for channel_index, (_, label, color, marker) in enumerate(CHANNELS):
    rgba = to_rgba(color)
    x_values = []
    y_values = []
    facecolors = []

    for temperature, filling, means, stds in points:
        alpha = _opacity(means[channel_index], stds[channel_index])
        x_values.append(1.0 - filling + marker_offsets[channel_index])
        y_values.append(temperature)
        facecolors.append((*rgba[:3], alpha))

    ax.scatter(
        x_values,
        y_values,
        marker=marker,
        s=55,
        facecolors=facecolors,
        edgecolors=facecolors,
    )
    channel_handles.append(
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            color=color,
            label=label,
        )
    )

ax.plot(1-METZNER_TC[1], METZNER_TC[2], color="k")
ax.plot(1-METZNER_DWAVE[1], METZNER_DWAVE[2], color="k", linestyle=":")
ax.set_xlabel(r"$\delta = 1 - n$")
ax.set_ylabel(r"$T / t$")
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
ax.legend(
    handles=channel_handles
)

plt.show()