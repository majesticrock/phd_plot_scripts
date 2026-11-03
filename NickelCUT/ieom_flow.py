import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc

from load_flow_files import load_flow_file
from nickel_cut_parameters import FLOW_PARAMETERS
from Momentum import Momentum

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

data = load_flow_file(
    **FLOW_PARAMETERS,
    resume_num="",
    force_json=False,
    dense=False,
)

print("Loaded data was produced on", data["time"])

L = data["L"]
end_index = data["number_of_data_points"]
l_times = data["l_times"]

cmap = plt.get_cmap("inferno")
norm = mc.Normalize(
    l_times[0],
    l_times[end_index - 1],
)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# ------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=2, ncols=3,
    figsize=(14, 7.5),
    constrained_layout=True,
    sharey=True, sharex=True
)

n = L // 2
path = (
    [Momentum(L, n, y) for y in range(n, -1, -1)]
    + [Momentum(L, x, 0) for x in range(n - 1, -1, -1)]
    + [Momentum(L, index, index) for index in range(1, n + 1)]
)

for ax in axes[1]:
    ax.set_xlabel(r"$\mathbf{q}$")
    ax.set_xticks([0, n, 2 * n, 3 * n])
    ax.set_xticklabels([r"$\Gamma$", "X", "M", r"$\Gamma$"])

axes[0,0].set_title("SC")
axes[0,1].set_title("Charge")
axes[0,2].set_title("Spin")
axes[0,0].set_ylabel(r"$\chi_s (q)$")
axes[1,0].set_ylabel(r"$\chi_d (q)$")

ax.set_ylim(0, 2.5)

for i in range(0, end_index, 5):
    color = cmap(norm(l_times[i]))

    for j, label in enumerate(["sc", "charge", "spin"]):
        chi_s_values, chi_d_values = data[f"{label}_susceptibilities"][i].T
        axes[0, j].plot(np.arange(len(path)), chi_s_values, c=color)
        axes[1, j].plot(np.arange(len(path)), chi_d_values, c=color)

fig.colorbar(
    sm,
    ax=axes,
    label=r"$\ell \cdot t$",
)

fig_ev, (ax_ev, ax_deriv, ax_rod) = plt.subplots(ncols=3, figsize=(15, 5), layout="constrained", sharex=True)

evs = np.array([
    data["extremal_sc_eigenvalues"],
    data["extremal_charge_eigenvalues"],
    data["extremal_spin_eigenvalues"]
])

ax_ev.plot(l_times, evs[0], ls="-",  marker="o", label="SC")
ax_ev.plot(l_times, evs[1], ls="--", marker="o", label="Charge")
ax_ev.plot(l_times, evs[2], ls="-.", marker="o", label="Spin")
ax_ev.plot(l_times, np.sqrt(evs[0]**2 + evs[1]**2 + evs[2]**2), ls=":", marker="o", c="k", label="Total")

ax_ev.set_xlabel(r"$\ell \cdot t$")
ax_ev.set_ylabel(r"$\lambda_\mathrm{ex}$")
ax_ev.legend()

gradients = np.abs(np.array([np.gradient(channel_evs, l_times) for channel_evs in evs]))
ex_gradient = np.sqrt(gradients[0]**2 + gradients[1]**2 + gradients[2]**2)

ax_deriv.plot(l_times, gradients[0] / ex_gradient[0], ls="-",  marker="o", label="SC")
ax_deriv.plot(l_times, gradients[1] / ex_gradient[0], ls="--", marker="o", label="Charge")
ax_deriv.plot(l_times, gradients[2] / ex_gradient[0], ls="-.", marker="o", label="Spin")

ax_deriv.plot(l_times, ex_gradient / ex_gradient[0], ls=":", c="k", marker="o", label="Total")

ax_deriv.set_xlabel(r"$\ell \cdot t$")
ax_deriv.set_ylabel(r"$|\frac{\partial}{\partial \ell} \lambda_\mathrm{ex}| / |\frac{\partial}{\partial \ell} \lambda_\mathrm{ex}(0)|$")
ax_deriv.set_yscale("log")
#ax_ev.set_xscale("log")
#ax_deriv.set_ylim(0, None)

ax_rod.plot(l_times, data["residual_offdiagonalities"], "-o", label=r"$\mathcal{H}$")
ax_rod.plot(l_times, data["max_interactions"] / 10, "-s", label=r"$\max |U|$ / 10")
ax_rod.plot(l_times, data["derivative_residual_offdiagonalities"] / 10, "-x", label=r"$\partial_\ell \mathcal{H}$ / 10")
ax_rod.set_xlabel(r"$\ell \cdot t$")
ax_rod.set_ylabel(r"$\mathrm{ROD} / t$")

ax_rod.set_ylim(0, None)
ax_rod.set_xlim(0, l_times[-1])
ax_rod.legend()

plt.show()