import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from create_momentum_labels import create_momentum_labels
from load_full_flow_file import load_full_flow_file
from Momentum import Momentum, MomentumGrid, Q, Gamma
from scipy.interpolate import RegularGridInterpolator

data = load_full_flow_file("cpp/NickelCUT/build/test")
ELL_STEP = data["index_of_lowest_ROD"]
L = data["L"]
N = L * L

usage_data = data["flow_states"][ELL_STEP]

q = MomentumGrid(L)
p = MomentumGrid(L)

def DW_channel(V):
    return V[(-p).flat_pos()[:,None], q.flat_pos()[None,:], Q(L).pos]

def BCS_channel(V):
    return V[p.flat_pos()[:,None], (-p).flat_pos()[:,None], q - p] + V[q.flat_pos()[:,None], (-q).flat_pos()[:,None], (p - q)].T

channel = BCS_channel
interaction = channel(usage_data["interactions_differing_spin"]).reshape(L, L, L, L).transpose(1, 0, 3, 2)

k = np.linspace(-np.pi, np.pi, L, endpoint=False)
dispersion = usage_data["dispersion"].reshape(L, L)

# --- Periodic extension ---
k_ext = np.concatenate([k, [np.pi]])
disp_ext = np.pad(dispersion, ((0, 1), (0, 1)), mode="wrap")

interp_dispersion = RegularGridInterpolator(
    (k_ext, k_ext),
    disp_ext,
    method="cubic",
    bounds_error=False,
    fill_value=None,
)

interaction_ext = np.pad(
    interaction,
    ((0, 1), (0, 1), (0, 1), (0, 1)),
    mode="wrap"
)
interp_interaction = RegularGridInterpolator(
    (k_ext, k_ext, k_ext, k_ext),
    interaction_ext,
    method="cubic",
    bounds_error=False,
    fill_value=None,
)

def wrap(k):
    return ((k + np.pi) % (2*np.pi)) - np.pi

# Checks on a 100x100 grid
L_inter = 50
q = np.linspace(-np.pi, np.pi, L_inter)
X, Y = np.meshgrid(q, q, indexing="xy")
points_2d = np.stack((wrap(X), wrap(Y)), axis=-1)
interpolated_dispersion = interp_dispersion(points_2d)

PX, PY, QX, QY = np.meshgrid(q, q, q, q, indexing="xy")
points_4d = np.stack((wrap(PX), wrap(PY), wrap(QX), wrap(QY)), axis=-1)
interpolated_interaction = interp_interaction(points_4d)

Deltas = 0.1 * np.ones(L_inter*L_inter)
Deltas_new = np.zeros(L_inter*L_inter)
error = 100.

while error > 1e-4:
    for ix in range(L_inter):
        for iy in range(L_inter):
            Deltas_new[ix + L_inter * iy] = -0.5 * np.sum(interpolated_interaction[ix, iy].flatten() * Deltas / np.sqrt(interpolated_dispersion.flatten()**2 + Deltas**2))
    error = np.linalg.norm(Deltas - Deltas_new)
    Deltas = Deltas_new.copy()
    print(f"Error = {error},  Delta_max = {np.max(np.abs(Deltas))}")

fig_sc, ax_sc = plt.subplots()
delta_im = ax_sc.imshow(Deltas.reshape(L_inter, L_inter), 
            extent=[-np.pi, np.pi, -np.pi, np.pi],
            origin="lower",
            aspect="equal")
cbar = fig_sc.colorbar(delta_im, ax=ax_sc)
ax_sc.set_xlabel(r"$k_x$")
ax_sc.set_ylabel(r"$k_y$")
cbar.set_label(r"$\Delta_\mathrm{SC}$")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
KX, KY = np.meshgrid(k, k, indexing="xy")
ax.scatter(KX.ravel(), KY.ravel(), dispersion.ravel(), s=60, c='k', label='data')
ax.plot_wireframe(X, Y, interpolated_dispersion, rstride=5, cstride=5,
                  color="m", alpha=0.5)


#points_4d = np.stack((
#    np.full_like(X, wrap(0.0)),      # k1x
#    np.full_like(X, wrap(-np.pi)),   # k1y
#    wrap(X),                         # k2x
#    wrap(Y),                         # k2y
#), axis=-1)
#Z = interp4(points_4d)
#
#vmax = np.max(np.abs(interaction))
#if vmax == 0.0:
#    vmax += 0.1
#norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
#
#fig, ax = plt.subplots()
#im = ax.imshow(
#    Z,
#    extent=[-np.pi, np.pi, -np.pi, np.pi],
#    origin="lower",
#    aspect="equal",
#    cmap="seismic",
#    norm=norm
#)
#fig.colorbar(im, ax=ax)
plt.show()