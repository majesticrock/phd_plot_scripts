
import numpy as np
import matplotlib.pyplot as plt

from create_momentum_labels import create_momentum_labels
from load_flow_files import load_flow_file
from nickel_cut_parameters import FLOW_PARAMETERS


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

ELL_INDEX = -1
TEMPERATURE = 0.1

# Set to 2 if the dispersion represents a spin-degenerate band
SPIN_DEGENERACY = 2

data = load_flow_file(
    **FLOW_PARAMETERS,
    resume_num="",
    force_json=False,
    dense=True,
)

L = data["L"]
channels = data["extracted_channels"]
n_steps = len(channels)

# ------------------------------------------------------------
# Dispersion at every flow step
# ------------------------------------------------------------

# Expected dispersion shape per step: (L*L,)
# Reshape to (n_steps, L, L).

dispersions = np.stack([
    np.asarray(channel["dispersion"]).reshape(L, L)
    for channel in channels
])

# ------------------------------------------------------------
# Calculate occupation numbers from the dispersion
# ------------------------------------------------------------

def fermi_occupation(energy, temperature):
    """Fermi-Dirac occupation, evaluated stably."""
    if temperature < 0:
        raise ValueError("Temperature must be non-negative.")

    if temperature == 0:
        return np.heaviside(-energy, 0.5)

    exponent = np.clip(energy / temperature, -700, 700)
    return 1.0 / (1.0 + np.exp(exponent))


occupations = fermi_occupation(
    dispersions,
    TEMPERATURE,
)

# ------------------------------------------------------------
# Occupation change at selected flow step
# ------------------------------------------------------------

delta_n = occupations[ELL_INDEX] - occupations[0]

print("Selected flow index:", ELL_INDEX)
print("Maximum absolute occupation change:",
      np.max(np.abs(delta_n)))

# ------------------------------------------------------------
# Total filling at every flow step
# ------------------------------------------------------------

# Filling per lattice site, including spin degeneracy if applicable.
filling = SPIN_DEGENERACY * occupations.mean(axis=(1, 2))

flow = np.arange(n_steps)

print("Initial filling:", filling[0])
print("Final filling:", filling[-1])
print("Change in filling:", filling[-1] - filling[0])

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

ticks, labels = create_momentum_labels(L)

fig, (ax_occ, ax_fill) = plt.subplots(
    1, 2,
    figsize=(13, 5),
    constrained_layout=True,
)

# Left: change in occupation numbers in momentum space
ex = np.max(np.abs(delta_n))
if ex == 0:
    ex = 1e-15

im = ax_occ.imshow(
    delta_n,
    origin="lower",
    aspect="equal",
    interpolation="nearest",
    cmap="seismic",
    vmin=-ex,
    vmax=ex,
)

fig.colorbar(im, ax=ax_occ, label=r"$\Delta n_{\mathbf{k}}$")

ax_occ.set_title(rf"Occupation change at $\ell = {data['l_times'][ELL_INDEX]:.3f}/t$")
ax_occ.set_xlabel(r"$k_y$")
ax_occ.set_ylabel(r"$k_x$")
ax_occ.set_xticks(ticks)
ax_occ.set_xticklabels(labels)
ax_occ.set_yticks(ticks)
ax_occ.set_yticklabels(labels)

# Right: total filling versus flow step
ax_fill.plot(data["l_times"], filling, marker=".", markersize=4)
ax_fill.set_xlabel(r"$\ell \cdot t$")
ax_fill.set_ylabel(r"Total filling $n$")
ax_fill.set_title("Filling along the flow")
ax_fill.grid(True, alpha=0.3)

fig.suptitle(
    rf"$T={TEMPERATURE:g}$"
)

plt.show()
