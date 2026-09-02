import numpy as np
import matplotlib.pyplot as plt
from load_full_flow_file import load_full_flow_file
from scipy.interpolate import RegularGridInterpolator

data = load_full_flow_file(subdir="", 
                           L=6,
                           T=0,
                           U_0=-1,
                           tprime=0,
                           E_F=0.01,
                           force_json=False)

ELL_STEP = data["index_of_lowest_ROD"]
L = data["L"]
N = L * L

usage_data = data["extracted_channels"][ELL_STEP]

# density_wave_differing | density_wave_same
# single_particle_energy_differing | single_particle_energy_same
# superconductivity
CHANNEL = "superconductivity"

interaction = 0.5 * (usage_data[CHANNEL] + usage_data[CHANNEL].T)
interaction = interaction.reshape(L, L, L, L) * N

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

# Checks on a finer grid
L_inter = 40
q = np.linspace(-np.pi, np.pi, L_inter, endpoint=False)
X, Y = np.meshgrid(q, q, indexing="xy")
points_2d = np.stack((wrap(X), wrap(Y)), axis=-1)
interpolated_dispersion = interp_dispersion(points_2d)

TOL = 1e-12
print("Filling:", ((interpolated_dispersion < -TOL).sum() + 0.5 * (np.abs(interpolated_dispersion) < TOL).sum()) / (L_inter*L_inter))

PX, PY, QX, QY = np.meshgrid(q, q, q, q, indexing="xy")
points_4d = np.stack((wrap(PX), wrap(PY), wrap(QX), wrap(QY)), axis=-1)
interpolated_interaction = interp_interaction(points_4d)

Deltas = 0.01 * np.ones(L_inter*L_inter) + 0.1j * np.linspace(0.0, 0.1, L_inter*L_inter)
Deltas_new = np.zeros(L_inter*L_inter) + 0.0001j
error = 100.

while error > 1e-8:
    for ix in range(L_inter):
        for iy in range(L_inter):
            Deltas_new[ix + L_inter * iy] = -0.5 * np.sum(interpolated_interaction[ix, iy].flatten() * Deltas / np.sqrt(interpolated_dispersion.flatten()**2 + Deltas**2))
    Deltas_new /= L_inter*L_inter
    error = np.linalg.norm(Deltas - Deltas_new)
    Deltas = Deltas_new.copy()
    print(f"Error = {error},  Delta_max = {np.max(np.abs(Deltas))}")

fig_re, ax_re = plt.subplots()
ax_re.set_title("Real part")
image_re = ax_re.imshow(Deltas.real.reshape(L_inter, L_inter), 
            extent=[-np.pi, np.pi * ( 1. - 1. / L_inter), -np.pi, np.pi * ( 1. - 1. / L_inter)],
            origin="lower",
            aspect="equal")
cbar_re = fig_re.colorbar(image_re, ax=ax_re)
ax_re.set_xlabel(r"$k_x$")
ax_re.set_ylabel(r"$k_y$")
cbar_re.set_label(r"$\Delta_\mathrm{SC}$")

fig_im, ax_im = plt.subplots()
ax_im.set_title("Imaginary part")
image_im = ax_im.imshow(Deltas.imag.reshape(L_inter, L_inter), 
            extent=[-np.pi, np.pi * ( 1. - 1. / L_inter), -np.pi, np.pi * ( 1. - 1. / L_inter)],
            origin="lower",
            aspect="equal")
cbar_im = fig_im.colorbar(image_im, ax=ax_im)
ax_im.set_xlabel(r"$k_x$")
ax_im.set_ylabel(r"$k_y$")
cbar_im.set_label(r"$\Delta_\mathrm{SC}$")

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#KX, KY = np.meshgrid(k, k, indexing="xy")
#ax.scatter(KX.ravel(), KY.ravel(), dispersion.ravel(), s=60, c='k', label='data')
#ax.plot_wireframe(X, Y, interpolated_dispersion, rstride=5, cstride=5,
#                  color="m", alpha=0.5)


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