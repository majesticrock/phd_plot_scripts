import matplotlib.pyplot as plt
from load_flow_files import *
from nickel_cut_parameters import FLOW_PARAMETERS

data = load_flow_file(**FLOW_PARAMETERS, force_json=False, dense=True)

fig, ax = plt.subplots()

ax.plot(data["l_times"], data["residual_offdiagonalities"], "-o")
ax.plot(data["l_times"], data["max_interactions"], "-s")

ax.set_xlabel(r"$\ell \cdot t$")
ax.set_ylabel(r"$\mathrm{ROD} / t$")

fig.tight_layout()

plt.show()