import numpy as np
import matplotlib.pyplot as plt
from mrock.get_data import *

data_loader = DataLoader()

data = data_loader.load_panda_file("cpp/NickelCUT/build/test.json.gz")

L = data["L"]

# turns a L*L 1D array into a L x L 2D array
def convert_1d_to_2d(arr):
    return np.reshape(arr, (-1, L))

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

    ax.plot(s1, y1, c=f"C{i}")
    ax.plot(s2, y2, c=f"C{i}")
    ax.plot(s3, y3, c=f"C{i}")

    

ax.axvline(L//2, c="k", ls=":")    
ax.axvline(L, c="k", ls=":")

ax.set_xlabel("$k$")
ax.set_ylabel(r"$\varepsilon$")

ax.set_xticks([0, n, 2*n, 3*n])
ax.set_xticklabels([r'$\Gamma$', 'X', 'M', r'$\Gamma$'])

plt.show()