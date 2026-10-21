import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from load_flow_files import load_flow_file
from nickel_cut_parameters import FLOW_PARAMETERS

data = load_flow_file(
    **FLOW_PARAMETERS,
    resume_num="",
    force_json=False,
    dense=True,
)
print("Loaded data was produced on", data["time"])
L = data["L"]


# Turns an L*L 1D array into an L x L 2D array
def convert_1d_to_2d(arr):
    return np.reshape(arr, (-1, L))


cmap = plt.get_cmap("inferno")
end_index = data["number_of_data_points"] - 1

norm = mc.Normalize(
    data["l_times"][0],
    data["l_times"][end_index - 1],
)

# Create a scalar mappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

fig, ax = plt.subplots()

for i in range(end_index):
    dispersion = convert_1d_to_2d(
        data["extracted_channels"][i]["dispersion"]
    )
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

    # (-π, 0) -> (0, -π)
    s4 = np.arange(n + 1) + 3 * n
    y4 = dispersion[
        np.arange(0, n + 1),
        np.arange(n, -1, -1),
    ]

    color = cmap(norm(data["l_times"][i]))

    ax.plot(s1, y1, c=color)
    ax.plot(s2, y2, c=color)
    ax.plot(s3, y3, c=color)
    ax.plot(s4, y4, c=color)


# High-symmetry point separators
ax.axvline(L // 2, c="k", ls=":")
ax.axvline(L, c="k", ls=":")
ax.axvline(3 * L // 2, c="k", ls="-")

ax.set_xlabel("$k$")
ax.set_ylabel(r"$\varepsilon$")

ax.set_xticks([0, n, 2 * n, 3 * n, 4 * n])
ax.set_xticklabels([
    r"$\Gamma$",
    "X",
    "M",
    r"$\Gamma$",
    r"$(0, -\pi)$",
])

# Colorbar for ell
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r"$\ell$")

fig.tight_layout()
plt.show()