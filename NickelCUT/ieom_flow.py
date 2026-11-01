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

cmap = plt.get_cmap("inferno")

norm = mc.Normalize(
    data["l_times"][0],
    data["l_times"][end_index - 1],
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
    color = cmap(norm(data["l_times"][i]))

    for j, label in enumerate(["sc", "charge", "spin"]):
        chi_s_values, chi_d_values = data[f"{label}_susceptibilities"][i].T
        axes[0, j].plot(np.arange(len(path)), chi_s_values, c=color)
        axes[1, j].plot(np.arange(len(path)), chi_d_values, c=color)

fig.colorbar(
    sm,
    ax=axes,
    label=r"$\ell \cdot t$",
)

fig_ev, (ax_ev, ax_deriv) = plt.subplots(ncols=2, figsize=(10, 5), layout="constrained")


ax_ev.plot(data["l_times"], data["extremal_sc_eigenvalues"],     ls="-",  label="SC")
ax_ev.plot(data["l_times"], data["extremal_charge_eigenvalues"], ls="--", label="Charge")
ax_ev.plot(data["l_times"], data["extremal_spin_eigenvalues"],   ls="-.", label="Spin")
ax_ev.plot(data["l_times"], np.sqrt(
    data["extremal_sc_eigenvalues"] * data["extremal_sc_eigenvalues"]
    + data["extremal_charge_eigenvalues"] * data["extremal_charge_eigenvalues"]
     + data["extremal_spin_eigenvalues"] * data["extremal_spin_eigenvalues"]
), ls=":", c="k", label="Total")

ax_ev.set_xlabel(r"$\ell \cdot t$")
ax_ev.set_ylabel(r"$\lambda_\mathrm{ex}$")
ax_ev.legend()

ax_deriv.plot(data["l_times"], np.gradient(data["extremal_sc_eigenvalues"],     data["l_times"]),     ls="-",  label="SC")
ax_deriv.plot(data["l_times"], np.gradient(data["extremal_charge_eigenvalues"], data["l_times"]), ls="--", label="Charge")
ax_deriv.plot(data["l_times"], np.gradient(data["extremal_spin_eigenvalues"],   data["l_times"]),   ls="-.", label="Spin")

ax_deriv.plot(data["l_times"], np.gradient(np.sqrt(
    data["extremal_sc_eigenvalues"] * data["extremal_sc_eigenvalues"]
    + data["extremal_charge_eigenvalues"] * data["extremal_charge_eigenvalues"]
     + data["extremal_spin_eigenvalues"] * data["extremal_spin_eigenvalues"]
), data["l_times"]), ls=":", c="k", label="Total")

ax_deriv.set_xlabel(r"$\ell \cdot t$")
ax_deriv.set_ylabel(r"$\partial_\ell \lambda_\mathrm{ex}$")
#ax_deriv.set_ylim(-3, 3)

plt.show()