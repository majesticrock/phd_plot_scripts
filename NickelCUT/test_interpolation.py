import numpy as np
import matplotlib.pyplot as plt
from load_full_flow_file import load_full_flow_file
from scipy.interpolate import RegularGridInterpolator
from Momentum import Momentum

data = load_full_flow_file(subdir="", 
                           L=10,
                           T=0,
                           U_0=-1,
                           tprime=0,
                           E_F=0.01,
                           resume_num="",
                           force_json=False)

ELL_STEP = data["index_of_lowest_ROD"]
L = data["L"]
N = L * L

usage_data = data["extracted_channels"][ELL_STEP]

# density_wave_differing | density_wave_same
# single_particle_energy_differing | single_particle_energy_same
# superconductivity
CHANNEL = "superconductivity"

# tranpose is required because we save the momenta in [x+L*y] instead of [y+L*x]
interaction = usage_data[CHANNEL].reshape(L, L, L, L).transpose(1, 0, 3, 2) * N

k = np.linspace(-np.pi, np.pi, L, endpoint=False)
dispersion = usage_data["dispersion"].reshape(L, L).T

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
    method="linear",
    bounds_error=False,
    fill_value=None,
)

def wrap(k):
    return ((k + np.pi) % (2*np.pi)) - np.pi

# Checks on a finer grid
L_inter = 40
q = np.linspace(-np.pi, np.pi, L_inter, endpoint=False)
X, Y = np.meshgrid(q, q, indexing="ij")
points_2d = np.stack((wrap(X), wrap(Y)), axis=-1)
interpolated_dispersion = interp_dispersion(points_2d).flatten()

TOL = 1e-12
print("Filling:", ((interpolated_dispersion < -TOL).sum() + 0.5 * (np.abs(interpolated_dispersion) < TOL).sum()) / (L_inter*L_inter))

PX, PY, QX, QY = np.meshgrid(q, q, q, q, indexing="ij")
points_4d = np.stack((wrap(PX), wrap(PY), wrap(QX), wrap(QY)), axis=-1)
interpolated_interaction = interp_interaction(points_4d)

Deltas = 0.01 * np.ones(L_inter*L_inter)
Deltas_new = np.zeros(L_inter*L_inter)
error = 100.

while error > 1e-6:
    for ix in range(L_inter):
        for iy in range(L_inter):
            Deltas_new[ix + L_inter * iy] = -0.5 * np.sum(interpolated_interaction[ix, iy].flatten() * Deltas / np.sqrt(interpolated_dispersion**2 + Deltas**2))
    Deltas_new /= L_inter*L_inter
    error = np.linalg.norm(Deltas - Deltas_new)
    Deltas = Deltas_new.copy()
    print(f"Error = {error},  Delta_max = {np.max(np.abs(Deltas))}")

fig_re, ax_re = plt.subplots()
ax_re.set_title("Real part")
image_re = ax_re.imshow(Deltas.reshape(L_inter, L_inter), 
            extent=[-np.pi, np.pi * ( 1. - 1. / L_inter), -np.pi, np.pi * ( 1. - 1. / L_inter)],
            origin="lower",
            aspect="equal")
cbar_re = fig_re.colorbar(image_re, ax=ax_re)
ax_re.set_xlabel(r"$k_x$")
ax_re.set_ylabel(r"$k_y$")
cbar_re.set_label(r"$\Delta_\mathrm{SC}$")



P = Momentum(L, 0, L//2)
# Plot the interpolated interaction for one fixed incoming momentum.
interaction_px, interaction_py = [P.kx, P.ky]
QX_slice, QY_slice = np.meshgrid(q, q, indexing="ij")
points_4d_slice = np.stack(
    (
        np.full_like(QX_slice, interaction_px),
        np.full_like(QX_slice, interaction_py),
        wrap(QX_slice),
        wrap(QY_slice),
    ),
    axis=-1,
)
interpolated_interaction_slice = interp_interaction(points_4d_slice)

raw_interaction_slice = interaction[P.x, P.y]
raw_qx, raw_qy = np.meshgrid(k, k, indexing="ij")

fig_interaction = plt.figure()
ax_interaction = fig_interaction.add_subplot(projection="3d")
surface = ax_interaction.plot_surface(
    QX_slice,
    QY_slice,
    interpolated_interaction_slice,
    cmap="seismic",
    linewidth=0,
    antialiased=True,
    alpha=0.8,
)
ax_interaction.scatter(
    raw_qx,
    raw_qy,
    raw_interaction_slice,
    color="black",
    s=24,
    depthshade=False,
    label="Raw interaction",
)
ax_interaction.set_xlabel(r"$q_x$")
ax_interaction.set_ylabel(r"$q_y$")
ax_interaction.set_zlabel(r"$V(\mathbf{p},\mathbf{q})$")
ax_interaction.set_title(
    rf"Interpolated {CHANNEL}, $p = ({interaction_px:.3g}, {interaction_py:.3g})$"
)
fig_interaction.colorbar(surface, ax=ax_interaction, shrink=0.7, pad=0.1, label=r"$V(\mathbf{p},\mathbf{q})$")
fig_interaction.tight_layout()

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#KX, KY = np.meshgrid(k, k, indexing="ij")
#ax.scatter(KX.ravel(), KY.ravel(), dispersion.ravel(), s=60, c='k', label='data')
#ax.plot_wireframe(X, Y, interpolated_dispersion, rstride=5, cstride=5,
#                  color="m", alpha=0.5)

plt.show()