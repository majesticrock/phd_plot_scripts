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
                                  field_amplitude=0.1, photon_energy=1., 
                                  tau_diag=10, tau_offdiag=-1, t0=0))

#main_df = load_panda("HHG", f"{DIR}/quench_laser/PiFlux", "occupations.json.gz", 
#                     **hhg_params(T=300, E_F=250, v_F=1.5e6, band_width=275, 
#                                  field_amplitude=1.6, photon_energy=5.25, 
#                                  tau_diag=10, tau_offdiag=-1, t0=8))

# Meshgrid for surfaces
nx, nz = main_df["upper_band"][0].shape
x = np.linspace(0, np.pi, nx, endpoint=False)
z = np.linspace(0, np.pi, nz, endpoint=False)
X, Z = np.meshgrid(x, z, indexing='ij')

laser_function = np.array(main_df["laser_function"])
time_steps = np.arange(len(laser_function))

norm = plt.Normalize(vmin=-1, vmax=1)
cmap = cm.PiYG

from matplotlib import gridspec
fig = plt.figure(figsize=(14, 7))
gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1])
fig.subplots_adjust(left=0.03, right=0.97, wspace=0.15)

ax3d = fig.add_subplot(gs[0], projection='3d')
ax2d = fig.add_subplot(gs[1])

ax3d.set_xlabel('$k_x$')
ax3d.set_ylabel('$k_z$')
ax3d.set_zlabel('$E(k_x, \\pi / 2, k_z)$ (meV)')
ax3d.view_init(elev=63, azim=-18)

ax2d.plot(time_steps, laser_function, color='blue', label="$\\tilde{A}(t)$")
ax2d.plot(time_steps[:-1], -3 * np.diff(laser_function), color="k", ls="--", label="$\\tilde{E}(t)$")
vertical_line = ax2d.axvline(0, color='red', lw=2)
ax2d.set_xlabel('Frame')
ax2d.set_ylabel('Laser Function')
ax2d.set_title("Laser Function vs. Time")
ax2d.legend()

mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])
fig.colorbar(mappable, ax=ax3d, shrink=0.6, pad=0.1, label='Change in occupation')

# 5.889401182228545meV = photon energy
Y_surf = 275 * 5.889401182228545 * np.sqrt(np.cos(X)**2 + np.cos(Z)**2)
equilibrium_data = main_df["upper_band"][0]


def update(frame):
    for coll in ax3d.collections[:]:
        coll.remove()

    data = -equilibrium_data + main_df["upper_band"][frame]
    
    facecolors = cmap(norm(data))

    ax3d.plot_surface(X, Z, Y_surf, facecolors=facecolors,
                      rstride=1, cstride=1, linewidth=0, antialiased=False)

    ax3d.set_title(f"Frame {frame+1}/{len(main_df['upper_band'])}")

    vertical_line.set_xdata([frame])

ani = FuncAnimation(fig, update, frames=len(main_df["upper_band"]), repeat=True)
ani.save("change.gif", writer="pillow", fps=15)

#plt.show()
