
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from Momentum import Momentum
from nickel_cut_parameters import FLOW_PARAMETERS
from mrock.get_data import DataLoader

loader = DataLoader()

flags = [
    "negative_excitation_charge",
    "negative_excitation_sc",
    "negative_excitation_spin",
]

data = (
    loader.load_all(
        f"nickel_cut/L={FLOW_PARAMETERS['L']}",
        "ieom_susceptibility.json.gz",
    )
    .query(
        f"U_0 == {FLOW_PARAMETERS['U_0']} "
        f"& tprime == {FLOW_PARAMETERS['tprime']} "
        f"& E_F == {FLOW_PARAMETERS['E_F']} "
        #f"& not {flags[0]} & not {flags[1]} & not {flags[2]}"
        f"& T > 0.1"
    )
    .sort_values("T", ignore_index=True, ascending=False)
)

L = FLOW_PARAMETERS["L"]
n = L // 2

path = (
    [Momentum(L, n, y) for y in range(n, -1, -1)]
    + [Momentum(L, x, 0) for x in range(n - 1, -1, -1)]
    + [Momentum(L, index, index) for index in range(1, n + 1)]
)

# Create a shared temperature-to-color mapping
cmap = plt.get_cmap("viridis")
norm = mpl.colors.Normalize(
    vmin=data["T"].min(),
    vmax=data["T"].max(),
)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

fig, axes = plt.subplots(
    figsize=(14, 5),
    ncols=3,
    layout="constrained",
    sharey=True,
)

axes[0].set_ylim(0., 10.)

for _, row in data.iterrows():
    # Color both curves according to the row's temperature
    color = cmap(norm(row["T"]))

    for i, label in enumerate(["SC", "spin", "charge"]):
        axes[i].set_title(label)

        chi_s_values, chi_d_values = row[label].T

        axes[i].plot(
            np.arange(len(path)),
            chi_s_values,# + 0.25 * _,
            color=color,
            ls="-",
        )
        axes[i].plot(
            np.arange(len(path)),
            chi_d_values,# + 0.25 * _,
            color=color,
            ls="--",
        )

        axes[i].set_xticks([0, n, 2 * n, 3 * n])
        axes[i].set_xticklabels([
            r"$\Gamma$", "X", "M", r"$\Gamma$"
        ])
        axes[i].set_xlabel(r"$\mathbf{q}$")
        axes[i].grid(axis="x", linestyle=":")

axes[0].set_ylabel(
    r"$\chi_{\mathrm{IEOM}}(\mathbf{q})$"
)

# Line-style legend: solid = s-wave, dashed = d-wave
from matplotlib.lines import Line2D

style_legend = [
    Line2D([0], [0], color="black", ls="-", label="$s$-wave"),
    Line2D([0], [0], color="black", ls="--", label="$d$-wave"),
]
axes[-1].legend(handles=style_legend, loc="best")

# Shared temperature colorbar
cbar = fig.colorbar(
    sm,
    ax=axes,
    location="right",
    pad=0.02,
)
cbar.set_label(r"Temperature $T/t$")

plt.show()
