import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc

from load_flow_files import load_flow_file
from nickel_cut_parameters import FLOW_PARAMETERS
from hartree_fock import solve_hartree_fock
from create_momentum_labels import create_momentum_labels

data = load_flow_file(
    **FLOW_PARAMETERS,
    resume_num="",
    force_json=False,
    dense=True,
)

print("Loaded data was produced on", data["time"])

L = data["L"]
end_index = data["number_of_data_points"] - 1


def convert_1d_to_2d(arr):
    return np.reshape(arr, (L, L))


# ------------------------------------------------------------
# Color mapping for the flow parameter
# ------------------------------------------------------------

cmap = plt.get_cmap("inferno")

norm = mc.Normalize(
    data["l_times"][0],
    data["l_times"][end_index - 1],
)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


# ------------------------------------------------------------
# Create two panels
# ------------------------------------------------------------

fig, (ax_band, ax_map) = plt.subplots(
    1, 2,
    figsize=(13, 5),
    constrained_layout=True,
)

hf_chemical_potentials = []
ticks, labels = create_momentum_labels(L)

# ------------------------------------------------------------
# Calculate and plot the HF solution at each flow step
# ------------------------------------------------------------

for i in range(end_index):
    hf_dispersion, mu, sigma = solve_hartree_fock(
        data["extracted_channels"][i],
        beta=data["beta"],
        target_filling=data["filling"],
    )

    hf_chemical_potentials.append(mu)

    dispersion = convert_1d_to_2d(hf_dispersion)
    n = L // 2

    # Γ -> X
    s1 = np.arange(n + 1)
    y1 = dispersion[np.arange(n, -1, -1), n]

    # X -> M
    s2 = np.arange(n + 1) + n
    y2 = dispersion[0, np.arange(n, -1, -1)]

    # M -> Γ
    s3 = np.arange(n + 1) + 2 * n
    idx = np.arange(n + 1)
    y3 = dispersion[idx, idx]

    # (−π, 0) -> (0, −π)
    s4 = np.arange(n + 1) + 3 * n
    y4 = dispersion[
        np.arange(0, n + 1),
        np.arange(n, -1, -1),
    ]

    color = cmap(norm(data["l_times"][i]))

    ax_band.plot(s1, y1, c=color)
    ax_band.plot(s2, y2, c=color)
    ax_band.plot(s3, y3, c=color)
    ax_band.plot(s4, y4, c=color)

    # Show the full HF energy grid at the final flow step.
    if i == end_index - 1:
        disp_max = np.max(np.abs(dispersion))
        im = ax_map.imshow(
            dispersion,
            origin="lower",
            aspect="equal",
            interpolation="nearest",
            cmap="RdBu_r",
            vmax=disp_max,
            vmin=-disp_max
        )

        fig.colorbar(
            im,
            ax=ax_map,
            label=r"$\epsilon_{\mathbf{k}}^{\mathrm{HF}}-\mu$",
        )

        ax_map.set_title(
            rf"HF energies at $\ell={data['l_times'][i]:.3f}/t$"
        )
        ax_map.set_xlabel(r"$k_y$")
        ax_map.set_ylabel(r"$k_x$")
        ax_map.set_xticks(ticks)
        ax_map.set_xticklabels(labels)
        ax_map.set_yticks(ticks)
        ax_map.set_yticklabels(labels)


# ------------------------------------------------------------
# Format the band-structure panel
# ------------------------------------------------------------

ax_band.axvline(L // 2, c="k", ls=":")
ax_band.axvline(L, c="k", ls=":")
ax_band.axvline(3 * L // 2, c="k", ls="-")

ax_band.set_xlabel(r"$k$")
ax_band.set_ylabel(
    r"$\epsilon_{\mathbf{k}}^{\mathrm{HF}}-\mu$"
)

ax_band.set_xticks([0, n, 2 * n, 3 * n, 4 * n])
ax_band.set_xticklabels([
    r"$\Gamma$",
    "X",
    "M",
    r"$\Gamma$",
    r"$(0,-\pi)$",
])

fig.colorbar(sm, ax=ax_band, label=r"$\ell$")

plt.show()