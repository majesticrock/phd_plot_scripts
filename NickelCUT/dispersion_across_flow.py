import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.lines import Line2D

from load_flow_files import load_flow_file
from nickel_cut_parameters import FLOW_PARAMETERS
from hartree_fock import solve_hartree_fock
from create_momentum_labels import create_momentum_labels


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

data = load_flow_file(
    **FLOW_PARAMETERS,
    resume_num="",
    force_json=False,
    dense=True,
)

print("Loaded data was produced on", data["time"])

L = data["L"]
end_index = data["number_of_data_points"]


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

xi_initial = None
xi_final = None


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

    # HF dispersion relative to the chemical potential.
    # Assumes hf_dispersion is the unshifted HF energy.
    dispersion = convert_1d_to_2d(hf_dispersion)
    xi = dispersion - mu

    n = L // 2

    # Save initial and final energy grids
    if i == 0:
        xi_initial = xi.copy()

    if i == end_index - 1:
        xi_final = xi.copy()

    # --------------------------------------------------------
    # Gamma -> X
    # --------------------------------------------------------

    s1 = np.arange(n + 1)
    y1 = xi[np.arange(n, -1, -1), n]

    # --------------------------------------------------------
    # X -> M
    # --------------------------------------------------------

    s2 = np.arange(n + 1) + n
    y2 = xi[0, np.arange(n, -1, -1)]

    # --------------------------------------------------------
    # M -> Gamma
    # --------------------------------------------------------

    s3 = np.arange(n + 1) + 2 * n
    idx = np.arange(n + 1)
    y3 = xi[idx, idx]

    # --------------------------------------------------------
    # (-pi, 0) -> (0, -pi)
    # --------------------------------------------------------

    s4 = np.arange(n + 1) + 3 * n
    y4 = xi[
        np.arange(0, n + 1),
        np.arange(n, -1, -1),
    ]

    color = cmap(norm(data["l_times"][i]))

    ax_band.plot(s1, y1, c=color)
    ax_band.plot(s2, y2, c=color)
    ax_band.plot(s3, y3, c=color)
    ax_band.plot(s4, y4, c=color)


# ------------------------------------------------------------
# Shift the energy grids to center the BZ at zero
# ------------------------------------------------------------

shift = L // 2

xi_initial_shifted = np.roll(
    xi_initial, shift=(shift, shift), axis=(0, 1)
)
xi_final_shifted = np.roll(
    xi_final, shift=(shift, shift), axis=(0, 1)
)

# Grid spacing
dk = 2 * np.pi / L

# Coordinates of the shifted grid-point centers
k = -np.pi + dk * np.arange(L)

# Periodically pad the energy grids on all four sides
xi_initial_pad = np.pad(xi_initial_shifted, 1, mode="wrap")
xi_final_pad = np.pad(xi_final_shifted, 1, mode="wrap")

# Coordinates corresponding to the padded grids
k_pad = np.concatenate((
    [k[0] - dk],
    k,
    [k[-1] + dk],
))

KX_pad, KY_pad = np.meshgrid(k_pad, k_pad, indexing="ij")


# ------------------------------------------------------------
# Plot the final energy map, including periodic wrapping
# ------------------------------------------------------------

disp_max = np.max(np.abs(xi_final_shifted))

im = ax_map.imshow(
    xi_final_pad,
    origin="lower",
    extent=[
        k_pad[0] - dk / 2,
        k_pad[-1] + dk / 2,
        k_pad[0] - dk / 2,
        k_pad[-1] + dk / 2,
    ],
    aspect="equal",
    interpolation="nearest",
    cmap="RdBu_r",
    vmin=-disp_max,
    vmax=disp_max,
)

fig.colorbar(
    im,
    ax=ax_map,
    label=r"$\varepsilon_{\mathbf{k}}^{\mathrm{HF}}-\mu$",
)


# ------------------------------------------------------------
# Plot both Fermi surfaces on the same periodically padded grid
# ------------------------------------------------------------

ax_map.contour(
    KY_pad,
    KX_pad,
    xi_initial_pad,
    levels=[0],
    colors=["black"],
    linestyles=["--"],
    linewidths=1.8,
)

ax_map.contour(
    KY_pad,
    KX_pad,
    xi_final_pad,
    levels=[0],
    colors=["black"],
    linestyles=["-"],
    linewidths=1.8,
)


# ------------------------------------------------------------
# Format the BZ
# ------------------------------------------------------------

ax_map.set_xlim(-np.pi, np.pi)
ax_map.set_ylim(-np.pi, np.pi)
ax_map.set_aspect("equal")

bz_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
bz_labels = [
    r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"
]

ax_map.set_xticks(bz_ticks)
ax_map.set_xticklabels(bz_labels)
ax_map.set_yticks(bz_ticks)
ax_map.set_yticklabels(bz_labels)

ax_map.set_xlabel(r"$k_y$")
ax_map.set_ylabel(r"$k_x$")

ax_map.set_title(
    rf"HF energies at $\ell={data['l_times'][end_index - 1]:.3f}/t$"
)

legend_lines = [
    Line2D(
        [0], [0],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Initial FS",
    ),
    Line2D(
        [0], [0],
        color="black",
        linestyle="-",
        linewidth=1.8,
        label="Final FS",
    ),
]

ax_map.legend(handles=legend_lines, loc="best")


# ------------------------------------------------------------
# Format the band-structure panel
# ------------------------------------------------------------

ax_band.axvline(L // 2, c="k", ls=":")
ax_band.axvline(L, c="k", ls=":")
ax_band.axvline(3 * L // 2, c="k", ls="-")

ax_band.set_xlabel(r"$k$")
ax_band.set_ylabel(
    r"$\varepsilon_{\mathbf{k}}^{\mathrm{HF}}-\mu$"
)

ax_band.set_xticks([
    0,
    n,
    2 * n,
    3 * n,
    4 * n,
])

ax_band.set_xticklabels([
    r"$\Gamma$",
    "X",
    "M",
    r"$\Gamma$",
    r"$(0,-\pi)$",
])

fig.colorbar(
    sm,
    ax=ax_band,
    label=r"$\ell$",
)

plt.show()