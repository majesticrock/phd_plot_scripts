import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mrock.get_data import *
from create_momentum_labels import create_momentum_labels

data_loader = DataLoader()
data = data_loader.load_panda_file("cpp/NickelCUT/build/test.json.gz")
L = data["L"]
ELL_INDEX = data["index_of_lowest_ROD"]

# turns a L*L 1D array into a L x L 2D array
def convert_1d_to_2d(arr):
    return np.reshape(arr, (-1, L))

X, Y = np.meshgrid(np.arange(0, L), np.arange(0, L))
Z = convert_1d_to_2d(data["flow_states"][ELL_INDEX]["dispersion"])
norm = plt.Normalize(Z.min(), Z.max())
colors = cm.inferno(norm(Z))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

#ax.plot_surface(X, Y, dispersion, cmap="inferno")
surf = ax.plot_surface(X, Y, Z, facecolors=colors, shade=False, linewidth=4)
surf.set_facecolor((0,0,0,0))

# Set custom tick labels for momentum space
ticks, labels = create_momentum_labels(L)
ax.set_xticks(ticks)
ax.set_xticklabels(labels)
ax.set_yticks(ticks)
ax.set_yticklabels(labels)
ax.set_xlabel(r"$k_y$")
ax.set_ylabel(r"$k_x$")


im_show_kwargs = {
    "origin":         "lower",
    "aspect":         "equal",
    "interpolation" : "nearest",
    "cmap" :          "tab10"
}
fig2d, ax2d = plt.subplots()
im = ax2d.imshow(Z, **im_show_kwargs)
fig2d.colorbar(im)

# Set custom tick labels for momentum space
ticks, labels = create_momentum_labels(L)
ax2d.set_xticks(ticks)
ax2d.set_xticklabels(labels)
ax2d.set_yticks(ticks)
ax2d.set_yticklabels(labels)
ax2d.set_xlabel(r"$k_y$")
ax2d.set_ylabel(r"$k_x$")


disp_flat = np.sort(Z.ravel())
print(disp_flat)

plt.show()