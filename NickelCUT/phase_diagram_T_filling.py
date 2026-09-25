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
data = data.sort_values("E_F").reset_index(drop=True)

fig, ax = plt.subplots()
filling_by_fermi_energy = {}
boundaries_by_order = defaultdict(list)

colors = plt.get_cmap("tab10")
order_colors = {}

for _, row in data.iterrows():
    E_F = row["E_F"]
    transitions = row["transition_data"]
    transition_fillings = []

    for transition in transitions:
        x = 1 - 0.5 * (transition["upper_filling"] + transition["lower_filling"]) 
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

for order in order_colors:
    x_values = []
    y_values = []

    for _, row in data.iterrows():
        matching_transitions = [
            transition for transition in row["transition_data"]
            if transition["order"] == order
        ]

        if matching_transitions:
            x = np.mean([
                0.5 * (transition["upper_filling"] + transition["lower_filling"])
                for transition in matching_transitions
            ])
            y = np.mean([
                0.5 * (transition["upper_temperature"] + transition["lower_temperature"])
                for transition in matching_transitions
            ])
        else:
            x = row["filling"]
            y = 0.0

        x_values.append(1-x)
        y_values.append(y)

    ax.plot(x_values, y_values, color=order_colors[order], linewidth=2)

ax.set_xlabel(r"$\delta$")
ax.set_ylabel(r"$T / t$")

#top_ax = ax.twiny()
#top_ax.set_xlim(ax.get_xlim())
#top_ax.set_xticks(list(filling_by_fermi_energy.values()))
#top_ax.set_xticklabels([f"{E_F:g}" for E_F in filling_by_fermi_energy])
#top_ax.set_xlabel(r"$E_F$")

ax.legend()

plt.show()