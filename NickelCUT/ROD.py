import matplotlib.pyplot as plt
from load_full_flow_file import load_full_flow_file

data = load_full_flow_file("cpp/NickelCUT/build/test")

fig, ax = plt.subplots()

ax.plot(data["l_times"], data["residual_offdiagonalities"])

ax.axvline(data["l_times"][data["index_of_lowest_ROD"]], ls=":", c="k")

ax.set_xlabel(r"$\ell \cdot t$")
ax.set_ylabel(r"$\mathrm{ROD} / t$")

fig.tight_layout()

plt.show()