import numpy as np
import matplotlib.pyplot as plt

from create_momentum_labels import create_momentum_labels
from load_flow_files import load_flow_file
from nickel_cut_parameters import FLOW_PARAMETERS
from hartree_fock import solve_hartree_fock

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

ELL_INDEX = -1
TEMPERATURE = FLOW_PARAMETERS["T"]
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
# Raw dispersions: used for the would-be filling
# ------------------------------------------------------------

dispersions = np.stack([
    np.asarray(channel["dispersion"]).reshape(L, L)
    for channel in channels
])

# ------------------------------------------------------------
# Hartree-Fock dispersions: used only for occupation changes
# ------------------------------------------------------------

hf_dispersions = []

for channel in channels:
    hf_dispersion, mu, sigma = solve_hartree_fock(
        channel,
        beta=1.0 / TEMPERATURE,
        target_filling=data["filling"],
    )

    hf_dispersions.append(
        np.asarray(hf_dispersion).reshape(L, L)
    )

hf_dispersions = np.stack(hf_dispersions)

# ------------------------------------------------------------
# Fermi-Dirac occupations
# ------------------------------------------------------------

def fermi_occupation(energy, temperature):
    """Fermi-Dirac occupation, evaluated stably."""
    if temperature < 0:
        raise ValueError("Temperature must be non-negative.")

    if temperature == 0:
        return np.heaviside(-energy, 0.5)

    exponent = np.clip(energy / temperature, -700, 700)
    return 1.0 / (1.0 + np.exp(exponent))


# Left-hand plot: HF occupations
hf_occupations = fermi_occupation(
    hf_dispersions,
    TEMPERATURE,
)

delta_n = hf_occupations[ELL_INDEX] - hf_occupations[0]

print("Selected flow index:", ELL_INDEX)
print(
    "Maximum absolute HF occupation change:",
    np.max(np.abs(delta_n)),
)

# Filling of the self-consistent Hartree-Fock solution
hf_filling = SPIN_DEGENERACY * hf_occupations.mean(axis=(1, 2))

print("Initial HF filling:", hf_filling[0])
print("Final HF filling:", hf_filling[-1])
print("Change in HF filling:", hf_filling[-1] - hf_filling[0])

# ------------------------------------------------------------
# Right-hand plot: would-be filling from raw dispersions
# ------------------------------------------------------------

raw_occupations = fermi_occupation(
    dispersions,
    TEMPERATURE,
)

filling = SPIN_DEGENERACY * raw_occupations.mean(axis=(1, 2))

print("Initial would-be filling:", filling[0])
print("Final would-be filling:", filling[-1])
print("Change in would-be filling:", filling[-1] - filling[0])

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

ticks, labels = create_momentum_labels(L)

fig, (ax_occ, ax_fill) = plt.subplots(
    1, 2,
    figsize=(13, 5),
    constrained_layout=True,
)

# Left: change in HF occupation numbers
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

fig.colorbar(
    im,
    ax=ax_occ,
    label=r"$\Delta n_{\mathbf{k}}^{\mathrm{HF}}$",
)

ax_occ.set_title(
    rf"HF occupation change at "
    rf"$\ell = {data['l_times'][ELL_INDEX]:.3f}/t$"
)
ax_occ.set_xlabel(r"$k_y$")
ax_occ.set_ylabel(r"$k_x$")
ax_occ.set_xticks(ticks)
ax_occ.set_xticklabels(labels)
ax_occ.set_yticks(ticks)
ax_occ.set_yticklabels(labels)

# Right: compare raw-dispersion and Hartree-Fock fillings
ax_fill.plot(
    data["l_times"],
    filling,
    marker=".",
    markersize=4,
    label="Raw dispersion",
)

ax_fill.plot(
    data["l_times"],
    hf_filling,
    marker=".",
    markersize=4,
    label="Hartree-Fock",
)

ax_fill.axhline(
    data["filling"],
    color="k",
    linestyle="--",
    linewidth=1,
    label="Target filling",
)

ax_fill.set_xlabel(r"$\ell \cdot t$")
ax_fill.set_ylabel(r"Filling $n$")
ax_fill.set_title("Filling along the flow")
ax_fill.grid(True, alpha=0.3)
ax_fill.legend()

fig.suptitle(rf"$T={TEMPERATURE:g}$")

plt.show()