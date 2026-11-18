import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from load_flow_files import load_flow_file
from nickel_cut_parameters import FLOW_PARAMETERS
from create_momentum_labels import create_momentum_labels

from Momentum import Momentum


def _normalize_eigenvector(eigenvector):
    max_abs_value = np.max(np.abs(eigenvector))
    if not np.isfinite(max_abs_value) or max_abs_value == 0.0:
        raise ValueError("Cannot normalize an eigenvector with zero or non-finite values.")
    return eigenvector / max_abs_value


data = load_flow_file(
    **FLOW_PARAMETERS,
    resume_num="",
    force_json=False,
    dense=False,
)
L = data["L"]
ELL_IDX = -1
fig, axes = plt.subplots(figsize=(12, 5), ncols=3, layout="constrained", sharex=True, sharey=True)

ticks, labels = create_momentum_labels(L)
for ax in axes:
    ax.set_xticks(ticks, labels)
axes[0].set_yticks(ticks, labels)

color_norm = Normalize(vmin=-1.0, vmax=1.0)
for ax, what in zip(axes, ["sc", "charge", "spin"]):
    transfer_momentum = Momentum(L, data[f'extremal_{what}_momenta'][ELL_IDX][0], data[f'extremal_{what}_momenta'][ELL_IDX][1])
    eigenvalue = data[f"extremal_{what}_eigenvalues"][ELL_IDX]
    eigenvector = data[f"extremal_{what}_eigenvectors"][ELL_IDX].reshape(L, L)
    eigenvector = _normalize_eigenvector(eigenvector)
    ax.set_title(f"{what}\n$q={transfer_momentum.to_string()}$ - $\\lambda={eigenvalue:.4f}$")
    
    im = ax.imshow(eigenvector, aspect="equal", cmap="viridis", norm=color_norm, origin="lower")

cbar = fig.colorbar(im, ax=axes)

plt.show()