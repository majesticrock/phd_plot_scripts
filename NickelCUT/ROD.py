import matplotlib.pyplot as plt
from load_full_flow_file import *
from nickel_cut_parameters import FLOW_PARAMETERS

data = load_all_resumed_files(**FLOW_PARAMETERS, force_json=False)

fig, ax = plt.subplots()

l0 = 0.
for i in range(len(data)):
    ax.plot(l0 + data[i]["l_times"], data[i]["residual_offdiagonalities"], "-o")
    ax.axvline(l0 + data[i]["l_times"][data[i]["index_of_lowest_ROD"]], ls=":", c="k", alpha=0.6)
    l0 += data[i]["l_times"][-1]

ax.set_xlabel(r"$\ell \cdot t$")
ax.set_ylabel(r"$\mathrm{ROD} / t$")

fig.tight_layout()

plt.show()