import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from mrock.get_data import DataLoader
from load_flow_files import load_all_resumed_files
from nickel_cut_parameters import FLOW_PARAMETERS


FLOW_FILE_PATTERN = "flow.json.gz*"
FLOW_FILE_RE = re.compile(r"^flow\.json\.gz\d*$")


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


available_temperatures = find_available_temperatures(FLOW_PARAMETERS)

if not available_temperatures:
    raise RuntimeError("No flow data found for the selected parameters.")

cmap = plt.get_cmap("viridis")
norm = Normalize(
    vmin=min(available_temperatures),
    vmax=max(available_temperatures),
)

fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")

for temperature in available_temperatures:
    parameters = {
        **FLOW_PARAMETERS,
        "T": temperature,
    }

    data = load_all_resumed_files(**parameters, force_json=False)
    color = cmap(norm(temperature))
    l0 = 0.0

    for segment_index, flow_data in enumerate(data):
        l_times = flow_data["l_times"]
        rod = flow_data["residual_offdiagonalities"]

        ax.plot(
            l0 + l_times,
            rod,
            "-o",
            color=color,
            label=rf"$T={temperature:g}$" if segment_index == 0 else None,
        )

        l0 += l_times[-1]

ax.set_xlabel(r"$\ell \cdot t$")
ax.set_ylabel(r"$\mathrm{ROD} / t$")
ax.legend()

ax.set_ylim(0.2, 2)

plt.show()