import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from mrock.get_data import DataLoader, nickel_cut_params
from load_flow_files import load_flow_file
from nickel_cut_parameters import FLOW_PARAMETERS

dense = True

FLOW_FILE_PATTERN = f"{'dense_' if dense else ''}flow.json.gz*"
FLOW_FILE_RE = re.compile(fr"^{FLOW_FILE_PATTERN}\d*$")
FILLING_DIR_RE = re.compile(r"^filling=(.+)$")


def find_available_fillings(parameters):
    loader = DataLoader()

    fixed_parameters = nickel_cut_params(
        parameters["L"],
        parameters["T"],
        parameters["U_0"],
        parameters["tprime"],
        parameters["filling"],
    )
    fixed_parameters.pop("filling")

    base_directory = loader._to_path(
        "nickel_cut",
        parameters["subdir"],
        **fixed_parameters,
    )

    relative_base = base_directory.relative_to(loader.data_dir)
    flow_files = loader.get_all_files(relative_base, FLOW_FILE_PATTERN)

    available = set()

    for flow_file in flow_files:
        path = Path(flow_file)

        if not FLOW_FILE_RE.match(path.name):
            continue

        match = FILLING_DIR_RE.match(path.parent.name)
        if match:
            available.add(float(match.group(1)))

    return sorted(available)


available_fillings = find_available_fillings(FLOW_PARAMETERS)

if not available_fillings:
    raise RuntimeError("No flow data found for the selected parameters.")

cmap = plt.get_cmap("viridis")
norm = Normalize(vmin=min(available_fillings), vmax=max(available_fillings))

fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")

for filling in available_fillings:
    parameters = {
        **FLOW_PARAMETERS,
        "filling": filling,
    }

    data = load_flow_file(**parameters, force_json=False, dense=dense)
    color = cmap(norm(filling))
    l0 = 0.0

    l_times = data["l_times"]
    rod = data["max_interactions"]

    ax.plot(
        l0 + l_times,
        rod,
        "-o",
        color=color,
        label=rf"$n={filling:g}$",
    )

ax.set_xlabel(r"$\ell \cdot t$")
ax.set_ylabel(r"$\mathrm{ROD} / t$")
ax.legend()

#ax.set_ylim(0.2, 2)

plt.show()