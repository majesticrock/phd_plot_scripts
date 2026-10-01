import matplotlib.pyplot as plt
from load_flow_files import *
from nickel_cut_parameters import FLOW_PARAMETERS

data = load_flow_file(**FLOW_PARAMETERS, force_json=False)#load_all_resumed_files(**FLOW_PARAMETERS, force_json=False)

fig, ax = plt.subplots()

l0 = 0.
#for i in range(len(data)):
#    ax.plot(l0 + data[i]["l_times"], data[i]["residual_offdiagonalities"], "-o")
#    ax.plot(l0 + data[i]["l_times"], data[i]["max_interactions"], "-s")
#    l0 += data[i]["l_times"][-1]

ax.plot(l0 + data["l_times"], data["residual_offdiagonalities"], "-o")
ax.plot(l0 + data["l_times"], data["max_interactions"], "-s")

ax.set_xlabel(r"$\ell \cdot t$")
ax.set_ylabel(r"$\mathrm{ROD} / t$")

fig.tight_layout()

plt.show()