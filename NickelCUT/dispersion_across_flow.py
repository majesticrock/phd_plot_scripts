import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from load_full_flow_file import load_full_flow_file

data = load_full_flow_file("cpp/NickelCUT/build/test")

L = data["L"]

# turns a L*L 1D array into a L x L 2D array
def convert_1d_to_2d(arr):
    return np.reshape(arr, (-1, L))

cmap = plt.get_cmap("inferno")
norm = mc.Normalize(data["l_times"][0], data["l_times"][-1])

fig, ax = plt.subplots()
for i in range(data["number_of_data_points"]):
    dispersion = convert_1d_to_2d(data["flow_states"][i]["dispersion"])
    
    n = L // 2
    # Γ -> X
    s1 = np.arange(n + 1)
    y1 = dispersion[np.arange(n, -1, -1), n]

    # X -> M
    s2 = np.arange(n + 1) + n
    y2 = dispersion[0, np.arange(n, -1, -1)]

    # M -> Γ
    s3 = np.arange(n + 1) + 2*n
    idx = np.arange(n + 1)
    y3 = dispersion[idx, idx]

    # (-π, 0) -> (0, -π)
    s4 = np.arange(n + 1) + 3*n
    y4 = dispersion[np.arange(0, n + 1), np.arange(n, -1, -1)]

    ax.plot(s1, y1, c=cmap(norm(data["l_times"][i])))
    ax.plot(s2, y2, c=cmap(norm(data["l_times"][i])))
    ax.plot(s3, y3, c=cmap(norm(data["l_times"][i])))
    ax.plot(s4, y4, c=cmap(norm(data["l_times"][i])))

    

ax.axvline(L//2, c="k", ls=":")    
ax.axvline(L, c="k", ls=":")
ax.axvline(3*L//2, c="k", ls=":")

ax.set_xlabel("$k$")
ax.set_ylabel(r"$\varepsilon$")

ax.set_xticks([0, n, 2*n, 3*n, 4*n])
ax.set_xticklabels([r'$\Gamma$', 'X', 'M', r'$\Gamma$', r'$(0, -\pi)$'])

plt.show()