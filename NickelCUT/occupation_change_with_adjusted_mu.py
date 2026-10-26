import numpy as np
import matplotlib.pyplot as plt

from create_momentum_labels import create_momentum_labels
from nickel_cut_parameters import FLOW_PARAMETERS
from mrock.get_data import DataLoader, nickel_cut_params

loader = DataLoader()
def fermi_occupation(energy, temperature):
    """Fermi-Dirac occupation, evaluated stably."""
    if temperature < 0:
        raise ValueError("Temperature must be non-negative.")

    if temperature == 0:
        return np.heaviside(-energy, 0.5)

    exponent = np.clip(energy / temperature, -700, 700)
    return 1.0 / (1.0 + np.exp(exponent))


loader = DataLoader()
data = loader.load_panda("nickel_cut", ".", "ieom_susceptibility.json.gz", **nickel_cut_params(**FLOW_PARAMETERS))
L = data["L"]

occupation_hf = fermi_occupation(data["dispersion"], FLOW_PARAMETERS["T"]).reshape(L, L)
k = np.linspace(-np.pi, np.pi, L, endpoint=False)
cos_x, cos_y = np.meshgrid(np.cos(k), np.cos(k))

disperion_null = -2. * (
    (cos_x + cos_y) 
    + 2 * FLOW_PARAMETERS["tprime"] * cos_x * cos_y
) - FLOW_PARAMETERS["E_F"]
occupation_null = fermi_occupation(disperion_null, FLOW_PARAMETERS["T"])

delta_n = occupation_null - occupation_hf

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

ticks, labels = create_momentum_labels(L)

fig, ax_occ = plt.subplots(
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

ax_occ.set_title(f"$\\delta = {1-2*np.average(occupation_hf):.2f}$   "
                 f"$\sum_k \Delta n_k = {2 * np.sum(delta_n):.2f}$   "
                 f"$\sum_k |\Delta n_k| = {2 * np.sum(np.abs(delta_n)):.2f}$")

ax_occ.set_xlabel(r"$k_y$")
ax_occ.set_ylabel(r"$k_x$")
ax_occ.set_xticks(ticks)
ax_occ.set_xticklabels(labels)
ax_occ.set_yticks(ticks)
ax_occ.set_yticklabels(labels)

plt.show()
