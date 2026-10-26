import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nickel_cut_parameters import FLOW_PARAMETERS
from mrock.get_data import DataLoader

loader = DataLoader()

# mu, filling, T_c
metzner_tc = np.array([
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

# mu, filling, T_c
metzner_dwave = np.array([
    [-1.121018062397373, 0.786820, 0.07020134228187919],
    [-1.0996715927750411, 0.800015, 0.07221476510067114],
    [-1.05, 0.828544, 0.04751677852348993],
    [-1.000328407224959, 0.854771, 0.04228187919463086],
    [-0.9506568144499179, 0.879313, 0.0006711409395973034],
]).T

# Load and filter data
data = (
    loader.load_all(
        f"nickel_cut/L={FLOW_PARAMETERS['L']}",
        "ieom_susceptibility.json.gz",
    )
    .query(
        f"U_0 == {FLOW_PARAMETERS['U_0']} "
        f"& tprime == {FLOW_PARAMETERS['tprime']}"
    )
    .sort_values("T", ignore_index=True)
)

flags = [
    "negative_excitation_charge",
    "negative_excitation_sc",
    "negative_excitation_spin",
    "pomeranchuk_instability"
]

max_T = pd.DataFrame({
    flag: (
        data.loc[data[flag]]
        .groupby("E_F")["T"]
        .max()
        .reindex(sorted(data["E_F"].unique()), fill_value=-1)
    )
    for flag in flags
}).fillna(-1)

fig, ax = plt.subplots(figsize=(8, 5))

for flag in flags:
    valid = data.loc[data[flag]]

    # Row corresponding to the maximum T for each E_F
    idx = valid.groupby("E_F")["T"].idxmax()
    points = valid.loc[idx].set_index("E_F").reindex(max_T.index)

    ax.plot(
        1. - points["hartree_fock_filling"],
        max_T[flag],
        marker="o",
        label=flag.replace("negative_excitation_", "").replace("_", " ").capitalize(),
    )

# Metzner reference curves
ax.plot(1. - metzner_tc[1], metzner_tc[2], color="k")
ax.plot(1. - metzner_dwave[1], metzner_dwave[2], color="k", ls=":")

# Axis labels and formatting
ax.set_xlabel(r"$\delta = 1 - n$")
ax.set_ylabel(r"$T / t$")
ax.legend()
ax.set_ylim(0, 0.16)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()