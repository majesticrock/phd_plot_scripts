import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nickel_cut_parameters import FLOW_PARAMETERS
from mrock.get_data import DataLoader

loader = DataLoader()

# Load and filter data
data = (
    loader.load_all(
        f"nickel_cut/L={FLOW_PARAMETERS['L']}",
        "ieom_susceptibility.json.gz",
    )
    .query(
        f"E_F == {FLOW_PARAMETERS['E_F']} "
        f"& tprime == {FLOW_PARAMETERS['tprime']}"
    )
    .sort_values("T", ignore_index=True)
)

flags = [
    "negative_excitation_charge",
    "negative_excitation_sc",
    "negative_excitation_spin",
]

# Find the maximum T for each U_0 and flag.
# If a flag is never True for a given U_0, assign T = -1.
all_EF = sorted(data["U_0"].unique())

max_T = pd.DataFrame({
    flag: (
        data.loc[data[flag]]
        .groupby("U_0")["T"]
        .max()
        .reindex(all_EF, fill_value=-1)
    )
    for flag in flags
}).fillna(-1)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

for flag in flags:
    ax.plot(
        max_T.index,
        max_T[flag],
        marker="o",
        label=flag.replace("negative_excitation_", "").capitalize(),
    )

# Axis labels and formatting
ax.set_xlabel(r"$\mu / t$")
ax.set_ylabel(r"$T / t$")
ax.legend()
ax.set_ylim(0, 0.16)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()