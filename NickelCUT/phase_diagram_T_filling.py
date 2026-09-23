import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from mrock.get_data import DataLoader
from nickel_cut_parameters import FLOW_PARAMETERS

MEAN_FIELD_FILE_NAME = "mean_field_solution.json.gz"
data_loader = DataLoader()

data = data_loader.load_all(f"nickel_cut/L={FLOW_PARAMETERS['L']}"
                     + F"/T={FLOW_PARAMETERS['T']}"
                     + F"/U_0={FLOW_PARAMETERS['U_0']}"
                     + F"/tprime={FLOW_PARAMETERS['tprime']}",
                     MEAN_FIELD_FILE_NAME)

fig, ax = plt.subplots()
filling_by_fermi_energy = {}
boundaries_by_order = defaultdict(list)
empty_phase_rows = []

colors = plt.get_cmap("tab10")
order_colors = {}

for _, row in data.iterrows():
    E_F = row["E_F"]
    transitions = row["transition_data"]
    transition_fillings = []

    if len(transitions) == 0:
        empty_phase_rows.append((E_F, row["filling"]))
    
    for transition in transitions:
        x = 0.5 * (transition["upper_filling"] + transition["lower_filling"])
        xerr = 0.5 * abs(transition["upper_filling"] - transition["lower_filling"])
        y = 0.5 * (transition["upper_temperature"] + transition["lower_temperature"])
        yerr = 0.5 * abs(transition["upper_temperature"] - transition["lower_temperature"])
        order = transition["order"]
        first_order_point = order not in order_colors
        if order not in order_colors:
            order_colors[order] = colors(len(order_colors) % 10)
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            marker="s",
            color=order_colors[order],
            label=order if first_order_point else None,
        )
        boundaries_by_order[order].append((E_F, x, y))
        transition_fillings.append(x)

    if transition_fillings:
        filling_by_fermi_energy[E_F] = np.mean(transition_fillings)
    else:
        filling_by_fermi_energy[E_F] = row["filling"]

for order, boundary in boundaries_by_order.items():
    boundary.sort()
    boundary_x = [point[1] for point in boundary]
    boundary_y = [point[2] for point in boundary]

    lower_end = boundary[0][0]
    upper_end = boundary[-1][0]
    lower_candidates = [row for row in empty_phase_rows if row[0] < lower_end]
    upper_candidates = [row for row in empty_phase_rows if row[0] > upper_end]

    if lower_candidates:
        _, filling = min(lower_candidates, key=lambda row: abs(row[0] - lower_end))
        boundary_x.insert(0, filling)
        boundary_y.insert(0, 0.0)
    if upper_candidates:
        _, filling = min(upper_candidates, key=lambda row: abs(row[0] - upper_end))
        boundary_x.append(filling)
        boundary_y.append(0.0)

    ax.plot(boundary_x, boundary_y, color=order_colors[order])

ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$T / t$")

top_ax = ax.twiny()
top_ax.set_xlim(ax.get_xlim())
top_ax.set_xticks(list(filling_by_fermi_energy.values()))
top_ax.set_xticklabels([f"{E_F:g}" for E_F in filling_by_fermi_energy])
top_ax.set_xlabel(r"$E_F$")

ax.legend()

plt.show()