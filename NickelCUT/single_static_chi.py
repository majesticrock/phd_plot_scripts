import numpy as np
import matplotlib.pyplot as plt

from Momentum import Momentum
from create_momentum_labels import create_momentum_labels
from nickel_cut_parameters import FLOW_PARAMETERS
from mrock.get_data import DataLoader, nickel_cut_params
loader = DataLoader()

data = loader.load_panda("nickel_cut", ".", "ieom_susceptibility.json.gz", **nickel_cut_params(**FLOW_PARAMETERS))
L = data["L"]

X, Y = np.meshgrid(np.arange(0, L), np.arange(0, L))
Z = data["dispersion"].reshape(L,L)

im_show_kwargs = {
    "origin":         "lower",
    "aspect":         "equal",
    "interpolation" : "nearest",
    "cmap" :          "seismic"
}

ex = max(abs(np.min(Z)), abs(np.max(Z)))
fig_hf, ax_hf = plt.subplots()
im = ax_hf.imshow(Z, vmin=-ex, vmax=ex, **im_show_kwargs)
fig_hf.colorbar(im)

# Set custom tick labels for momentum space
ticks, labels = create_momentum_labels(L)
ax_hf.set_xticks(ticks)
ax_hf.set_xticklabels(labels)
ax_hf.set_yticks(ticks)
ax_hf.set_yticklabels(labels)
ax_hf.set_xlabel(r"$k_y$")
ax_hf.set_ylabel(r"$k_x$")


n = L // 2
path = (
    [Momentum(L, n, y) for y in range(n, -1, -1)]
    + [Momentum(L, x, 0) for x in range(n - 1, -1, -1)]
    + [Momentum(L, index, index) for index in range(1, n + 1)]
)

fig, axes = plt.subplots(figsize=(14, 5), ncols=3, layout="constrained", sharey=True)

for i, label in enumerate(["SC", "spin", "charge"]):
    chi_s_values, chi_d_values = data[label].T
    axes[i].plot(np.arange(len(path)), chi_s_values, "b-",  label=f"$s$-wave")
    axes[i].plot(np.arange(len(path)), chi_d_values, "r--", label=f"$d$-wave")

    axes[i].set_xticks([0, n, 2 * n, 3 * n])
    axes[i].set_xticklabels([r"$\Gamma$", "X", "M", r"$\Gamma$"])
    axes[i].set_xlabel(r"$\mathbf{q}$")
    axes[i].set_title(label)
    axes[i].grid(axis="x", linestyle=":")

axes[0].set_ylabel(r"$\chi_{\mathrm{IEOM}}(\mathbf{q})$")
axes[-1].legend()

plt.show()