import matplotlib.pyplot as plt
from load_full_flow_file import load_full_flow_file

data = load_full_flow_file(subdir="", 
                           L=6,
                           T=0,
                           U_0=-1,
                           tprime=0,
                           E_F=0.01,
                           force_json=False)

fig, ax = plt.subplots()

ax.plot(data["l_times"], data["residual_offdiagonalities"], "-o")

ax.axvline(data["l_times"][data["index_of_lowest_ROD"]], ls=":", c="k")

ax.set_xlabel(r"$\ell \cdot t$")
ax.set_ylabel(r"$\mathrm{ROD} / t$")

fig.tight_layout()

plt.show()