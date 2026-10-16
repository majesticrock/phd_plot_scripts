import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from mrock.get_data import DataLoader
from nickel_cut_parameters import FLOW_PARAMETERS

metzner_tc = np.array([
[-0.7450836820083683, 0.1517948717948718],
[-0.7989539748953975, 0.14991452991452991],
[-0.8308577405857741, 0.14991452991452991],
[-0.8998953974895398, 0.11504273504273504],
[-0.9506276150627615, 0.11452991452991454],
[-1.0003138075313807, 0.10974358974358975],
[-1.1498953974895398, 0.06444444444444444],
[-1.2006276150627615, 0.039487179487179475],
[-1.2524058577405857, 0.024444444444444435],
[-1.297907949790795, 0.014871794871794852],
[-1.349163179916318, 0.010256410256410248]]).T

metzner_dwave = np.array([
[-1.121018062397373, 0.07020134228187919],
[-1.0996715927750411, 0.07221476510067114],
[-1.05, 0.04751677852348993],
[-1.000328407224959, 0.04228187919463086],
[-0.9506568144499179, 0.0006711409395973034]
]).T

MEAN_FIELD_FILE_NAME = "mean_field_solution.json.gz"
data_loader = DataLoader()

subdir = f"/{FLOW_PARAMETERS['subdir']}" if FLOW_PARAMETERS['subdir'] != "" else ""
data = data_loader.load_all(f"nickel_cut{subdir}/L={FLOW_PARAMETERS['L']}",
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

ax.plot(-metzner_tc[0], metzner_tc[1], color="k")
ax.plot(-metzner_dwave[0], metzner_dwave[1], color="k", ls=":")

ax.set_xlabel(r"$\mu / t$")
ax.set_ylabel(r"$T / t$")
ax.set_ylim(0, None)
ax.legend()

plt.show()