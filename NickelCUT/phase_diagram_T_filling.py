import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from mrock.get_data import DataLoader
from nickel_cut_parameters import FLOW_PARAMETERS

MEAN_FIELD_FILE_NAME = "mean_field_solution.json.gz"
data_loader = DataLoader()

data = data_loader.load_all(f"nickel_cut/L={FLOW_PARAMETERS['L']}",
                     MEAN_FIELD_FILE_NAME, 
                     condition=[
                         f"U_0={FLOW_PARAMETERS['U_0']}", 
                        f"tprime={FLOW_PARAMETERS['tprime']}"
                    ])
data = data.sort_values("E_F").reset_index(drop=True)

ORDER_PARAMETERS = {
    "Delta_SC": "SC",
    "Delta_AFM": "AFM",
    "Delta_CDW": "CDW",
}
ORDER_PARAMETER_THRESHOLD = 1e-10
MARKERS = {
    (False, False, False): "x",
    (False, False, True): "v",
    (False, True, False): "s",
    (False, True, True): "D",
    (True, False, False): "o",
    (True, False, True): "^",
    (True, True, False): "P",
    (True, True, True): "*",
}
points_by_status = defaultdict(lambda: {"x": [], "y": []})

for _, row in data.iterrows():
    active_status = tuple(
        bool(
            np.any(
                np.abs(np.asarray(row[column], dtype=float))
                > ORDER_PARAMETER_THRESHOLD
            )
        )
        for column in ORDER_PARAMETERS
    )
    points_by_status[active_status]["x"].append(-row["E_F"])#1.0 - float(row["filling"]))
    points_by_status[active_status]["y"].append(float(row["T"]))

fig, ax = plt.subplots(layout="constrained")

for active_status, points in points_by_status.items():
    active_names = [
        name
        for name, is_active in zip(ORDER_PARAMETERS.values(), active_status)
        if is_active
    ]
    label = ", ".join(active_names) if active_names else "None"
    ax.scatter(
        points["x"],
        points["y"],
        marker=MARKERS[active_status],
        label=label,
    )

ax.set_xlabel(r"$\delta$")
ax.set_ylabel(r"$T / t$")
ax.legend()

plt.show()