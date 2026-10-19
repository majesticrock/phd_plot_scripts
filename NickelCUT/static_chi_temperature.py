import numpy as np
import matplotlib.pyplot as plt

from Momentum import Momentum
from nickel_cut_parameters import FLOW_PARAMETERS
from mrock.get_data import DataLoader
loader = DataLoader()


data = loader.load_all(f"nickel_cut/L={FLOW_PARAMETERS['L']}", "ieom_susceptibility.json.gz").query(
    f"U_0 == {FLOW_PARAMETERS['U_0']} & tprime == {FLOW_PARAMETERS['tprime']} & E_F == {FLOW_PARAMETERS['E_F']} & T > 0"
).sort_values("T", ignore_index=True)

L = FLOW_PARAMETERS["L"]
n = L // 2
path = (
    [Momentum(L, n, y) for y in range(n, -1, -1)]
    + [Momentum(L, x, 0) for x in range(n - 1, -1, -1)]
    + [Momentum(L, index, index) for index in range(1, n + 1)]
)

fig, axes = plt.subplots(figsize=(14, 5), ncols=3, layout="constrained", sharey=True)
axes[0].set_ylim(-0.1, 5)

for r, row in data.iterrows():
    for i, label in enumerate(["SC", "spin", "charge"]):
        if r==0:
            axes[i].set_title(label)
        chi_s_values, chi_d_values = row[label].T
        axes[i].plot(np.arange(len(path)), chi_s_values, "b-",  label=f"$s$-wave" if r==0 else "")
        axes[i].plot(np.arange(len(path)), chi_d_values, "r--", label=f"$d$-wave" if r==0 else "")

        axes[i].set_xticks([0, n, 2 * n, 3 * n])
        axes[i].set_xticklabels([r"$\Gamma$", "X", "M", r"$\Gamma$"])
        axes[i].set_xlabel(r"$\mathbf{q}$")
        axes[i].grid(axis="x", linestyle=":")

axes[0].set_ylabel(r"$\chi_{\mathrm{IEOM}}(\mathbf{q})$")
axes[-1].legend()

plt.show()