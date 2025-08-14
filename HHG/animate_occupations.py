import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

DIR = "test"

main_df = load_panda("HHG", f"{DIR}/exp_laser/PiFlux", "occupations.json.gz", 
                     **hhg_params(T=300, E_F=118, v_F=1.5e6, band_width=275, 
                                  field_amplitude=1., photon_energy=1., 
                                  tau_diag=10, tau_offdiag=-1, t0=0))

#main_df = load_panda("HHG", f"{DIR}/quench_laser/PiFlux", "occupations.json.gz", 
#                     **hhg_params(T=300, E_F=118, v_F=1.5e6, band_width=275, 
#                                  field_amplitude=1.6, photon_energy=5.25, 
#                                  tau_diag=10, tau_offdiag=-1, t0=8))

# Meshgrid for surfaces
nx, nz = main_df["upper_band"][0].shape
x = np.linspace(0, np.pi, nx)
z = np.linspace(-np.pi, np.pi, nz)
X, Z = np.meshgrid(x, z, indexing='ij')

# Laser function data
laser_function = np.array(main_df["laser_function"])
time_steps = np.arange(len(laser_function))

# Colormap settings
norm = plt.Normalize(vmin=0, vmax=1)
cmap = cm.viridis

# Figure with 2 panels
fig = plt.figure(figsize=(14, 7))
ax3d = fig.add_subplot(121, projection='3d')
ax2d = fig.add_subplot(122)

# 3D panel setup
ax3d.set_xlabel('$k_x$')
ax3d.set_ylabel('$k_z$')
ax3d.set_zlabel('$E(k_x, \\pi / 2, k_z)$ (meV)')
ax3d.view_init(elev=47, azim=-30)

# 2D laser function panel setup
ax2d.plot(time_steps, laser_function, color='blue')
vertical_line = ax2d.axvline(0, color='red', lw=2)
ax2d.set_xlabel('Frame')
ax2d.set_ylabel('Laser Function')
ax2d.set_title("Laser Function vs. Time")

# Colorbar
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])
fig.colorbar(mappable, ax=ax3d, shrink=0.6, pad=0.1, label='Occupation')

Y_surf = np.sqrt(np.cos(X)**2 + np.cos(Z)**2)
Y_surf_neg = -Y_surf

def shift_occupations_back(data, laser_shift):
    # data shape: (nx, nz)
    # z array goes from -π to π
    nz = data.shape[1]
    dz = (2 * np.pi) / nz
    shift_cols = int(round(laser_shift / dz))
    return np.roll(data, -shift_cols, axis=1)  # negative to undo


def update(frame):
    # Remove old surfaces
    for coll in ax3d.collections[:]:
        coll.remove()

    # Laser shift for this frame
    laser = laser_function[frame]

    # Undo shift for occupations
    data_top = shift_occupations_back(main_df["upper_band"][frame], laser)
    data_bottom = shift_occupations_back(main_df["lower_band"][frame], laser)

    # Facecolors for fixed dispersion
    facecolors_top = cmap(norm(data_top))
    facecolors_bottom = cmap(norm(data_bottom))

    # Plot fixed dispersion with updated colors
    ax3d.plot_surface(X, Z, Y_surf, facecolors=facecolors_top,
                      rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax3d.plot_surface(X, Z, Y_surf_neg, facecolors=facecolors_bottom,
                      rstride=1, cstride=1, linewidth=0, antialiased=False)

    ax3d.set_title(f"Frame {frame+1}/{len(main_df['upper_band'])}")

    # Move vertical line in laser panel
    vertical_line.set_xdata([frame])



# Animation
ani = FuncAnimation(fig, update, frames=len(main_df["upper_band"]),
                    interval=2, repeat=True)

ani.save("animation.gif", writer="pillow", fps=30)

plt.show()
