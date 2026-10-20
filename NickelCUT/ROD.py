import matplotlib.pyplot as plt
from load_flow_files import *
from nickel_cut_parameters import FLOW_PARAMETERS

data = load_flow_file(**FLOW_PARAMETERS, force_json=False, dense=True)

fig, ax = plt.subplots()

ax.plot(data["l_times"], data["residual_offdiagonalities"], "-o", label=r"$\mathcal{H}$")
ax.plot(data["l_times"], data["max_interactions"] / 10, "-s", label=r"$\max |U|$ / 10")
ax.plot(data["l_times"], data["derivative_residual_offdiagonalities"], "-x", label="$\partial_\ell \mathcal{H}$")

print("l_max =", data["l_times"][-1])

ax.set_xlabel(r"$\ell \cdot t$")
ax.set_ylabel(r"$\mathrm{ROD} / t$")

ax.set_ylim(0, None)
ax.set_xlim(0, data["l_times"][-1])
ax.legend()

fig.tight_layout()

plt.show()