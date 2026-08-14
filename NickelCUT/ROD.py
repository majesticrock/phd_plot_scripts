import numpy as np
import matplotlib.pyplot as plt
from mrock.get_data import *

data_loader = DataLoader()

data = data_loader.load_panda_file("cpp/NickelCUT/build/test.json.gz")

fig, ax = plt.subplots()

ax.plot(data["l_times"], data["residual_offdiagonalities"])

ax.axvline(data["l_times"][data["index_of_lowest_ROD"]], ls=":", c="k")

ax.set_xlabel(r"$\ell \cdot t$")
ax.set_ylabel(r"$\mathrm{ROD} / t$")

fig.tight_layout()

plt.show()