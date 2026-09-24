import numpy as np
import matplotlib.pyplot as plt
from load_full_flow_file import load_full_flow_state
from nickel_cut_parameters import FLOW_PARAMETERS
from copy import copy
from Momentum import Momentum

data = load_full_flow_state(**FLOW_PARAMETERS, resume_num="", force_json=False)
L = data["L"]

def extract_fermi_surface(dispersion):
    FS_points = []
    closest_to_zero_momentum = Momentum(L, 0, 0)
    
    for x in range((L//2+1)//2+1):
        closest_to_zero_value = 10.
        for y in range(L//2+1):
            k = Momentum(L, x, y)
            if (np.abs(dispersion[k.pos]) < closest_to_zero_value):
                closest_to_zero_value = np.abs(dispersion[k.pos])
                closest_to_zero_momentum = copy(k)
                
        FS_points.append(closest_to_zero_momentum)
    
    size = len(FS_points)
    for i in range((L//2)%2, size):
        FS_points.append(Momentum(L, FS_points[size-i-1].y, FS_points[size-i-1].x))
    
    size = len(FS_points)
    for i in range(size):
        rotated = Momentum(L, L-FS_points[i].y, FS_points[i].x)
        if (rotated not in FS_points):
            FS_points.append(rotated)
    for i in range(size):
        rotated = Momentum(L, L-FS_points[i].x, L-FS_points[i].y)
        if (rotated not in FS_points):
            FS_points.append(rotated)
    for i in range(size):
        rotated = Momentum(L, FS_points[i].y, L-FS_points[i].x)
        if (rotated not in FS_points):
            FS_points.append(rotated)
    
    return FS_points

FS_points = extract_fermi_surface(data["dispersion"])
FS_indices = np.array([point.pos for point in FS_points])
fix_k3 = Momentum(L, 0, L//2)
Q_indices = np.array([ (point - fix_k3).pos for point in FS_points ])

im_show_kwargs = {
    "origin":         "lower",
    "aspect":         "equal",
    "interpolation" : "nearest",
    "cmap" :          "seismic"
}

fig, (ax_fs, ax) = plt.subplots(ncols=2, figsize=(12, 5), layout="constrained")

perm = np.array([
    ((-x) % L) + L*((-y) % L)
    for y in range(L)
    for x in range(L)
])

V = 2 * data["interactions_differing_spin"][FS_indices[None,:], FS_indices[:,None], Q_indices[:,None]] * L*L
vmax = np.max(np.abs(V))
if vmax == 0.0:
    vmax += 0.1
im = ax.imshow(V, vmin=-vmax, vmax=vmax, **im_show_kwargs)

ax.set_xlabel(r"$p_i$")
ax.set_ylabel(r"$k_i$")
fig.colorbar(im, ax=ax, label=r"$U(\mathbf{k},\mathbf{p},\mathbf{q})$")

dispersion_2d = data["dispersion"].reshape(L, L)
dispersion_vmax = np.max(np.abs(dispersion_2d))
if dispersion_vmax == 0.0:
    dispersion_vmax = 0.1
half_grid_step = np.pi / L
im_fs = ax_fs.imshow(
    dispersion_2d,
    extent=(
        -np.pi - half_grid_step,
        np.pi - half_grid_step,
        -np.pi - half_grid_step,
        np.pi - half_grid_step,
    ),
    vmin=-dispersion_vmax,
    vmax=dispersion_vmax,
    **im_show_kwargs,
)
fig.colorbar(im_fs, ax=ax_fs, label=r"$\varepsilon(\mathbf{k})$")
ax_fs.scatter(
    [point.kx for point in FS_points],
    [point.ky for point in FS_points],
    color="black",
    zorder=2,
)
for number, point in enumerate(FS_points):
    ax_fs.annotate(
        str(number),
        (point.kx, point.ky),
        xytext=(4, 4),
        textcoords="offset points",
    )
ax_fs.set_xlabel(r"$k_x$")
ax_fs.set_ylabel(r"$k_y$")
ax_fs.set_aspect("equal")

plt.show()