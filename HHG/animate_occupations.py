import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

BAND_WIDTH=300
MOD_FLUX = False
DIR = "test"
main_df = load_panda("HHG", f"{DIR}/powerlaw1_laser/{'ModifiedPiFlux' if MOD_FLUX else 'PiFlux'}", "occupations.json.gz", 
                     **hhg_params(T=300, E_F=118, v_F=1.5e6, band_width=BAND_WIDTH, 
                                  field_amplitude=1.6, photon_energy=5.25, 
                                  tau_diag=15, tau_offdiag=-1, t0=8))

#DIR = "test"
#main_df = load_panda("HHG", f"{DIR}/quench_laser/PiFlux", "occupations.json.gz", 
#                     **hhg_params(T=300, E_F=250, v_F=1.5e6, band_width=BAND_WIDTH, 
#                                  field_amplitude=1.6, photon_energy=5.25, 
#                                  tau_diag=10, tau_offdiag=-1, t0=8))

#main_df = load_panda("HHG", f"{DIR}/powerlaw1_laser/PiFlux", "occupations.json.gz", 
#                     **hhg_params(T=300, E_F=250, v_F=1.5e6, band_width=BAND_WIDTH, 
#                                  field_amplitude=1.6, photon_energy=5.25, 
#                                  tau_diag=15, tau_offdiag=-1, t0=8))

# Meshgrid for surfaces
nx, nz = main_df["upper_band"][0].shape
x = np.linspace(-np.pi if MOD_FLUX else 0, np.pi, nx, endpoint=False)
z = np.linspace(-np.pi , np.pi, nz, endpoint=False)
X, Z = np.meshgrid(x, z, indexing='ij')

laser_function = np.array(main_df["laser_function"])
time_steps = np.arange(len(laser_function))

norm = plt.Normalize(vmin=0, vmax=1)
cmap = cm.viridis

from matplotlib import gridspec
fig = plt.figure(figsize=(14, 7))
gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1])
fig.subplots_adjust(left=0.03, right=0.97, wspace=0.15)

ax3d = fig.add_subplot(gs[0], projection='3d')
ax2d = fig.add_subplot(gs[1])

ax3d.set_xlabel('$k_x$', labelpad=15)
ax3d.set_ylabel('$k_z(t)$', labelpad=15)
ax3d.set_zlabel('$E(k_x, \\pi / 2, k_z(t))$ (eV)', labelpad=15)
ax3d.view_init(elev=47, azim=-30)

import matplotlib.ticker as ticker
def pi_formatter(x, pos):
    frac = x / np.pi
    if np.isclose(frac, 0):
        return "0"
    elif np.isclose(frac, 1):
        return r"$\pi$"
    elif np.isclose(frac, 0.5):
        return r"$\pi/2$"
    elif np.isclose(frac, 0.25):
        return r"$\pi/4$"
    elif np.isclose(frac, 0.75):
        return r"$3\pi/4$"
    else:
        return f"{frac:.2f}$\pi$"
    
ax3d.xaxis.set_major_locator(ticker.MultipleLocator(np.pi/2))
ax3d.xaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))
ax3d.yaxis.set_major_locator(ticker.MultipleLocator(np.pi/2))
ax3d.yaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))

ax2d.plot(time_steps, laser_function, color='blue', label="$\\tilde{A}(t)$")
ax2d.plot(time_steps[:-1], -3 * np.diff(laser_function), color="k", ls="--", label="$\\tilde{E}(t)$")
vertical_line = ax2d.axvline(0, color='red', lw=2)
ax2d.set_xlabel('Frame')
ax2d.set_ylabel('Laser Function')
ax2d.set_title("Laser Function vs. Time")
ax2d.legend()

mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])
fig.colorbar(mappable, ax=ax3d, shrink=0.6, pad=0.1, label='Occupation')

# 5.889401182228545meV = photon energy
Y_surf = 1e-3 * (2. / np.sqrt(12.)) * BAND_WIDTH * 5.889401182228545 * np.sqrt((1. - np.cos(X))**2 + (1. - np.cos(Z))**2 if MOD_FLUX else np.cos(X)**2 + np.cos(Z)**2)
Y_surf_neg = -Y_surf

def update(frame):
    for coll in ax3d.collections[:]:
        coll.remove()

    data_top = main_df["upper_band"][frame]
    data_bottom = main_df["lower_band"][frame]
    facecolors_top = cmap(norm(data_top))
    facecolors_bottom = cmap(norm(data_bottom))

    ax3d.plot_surface(X, Z, Y_surf, facecolors=facecolors_top,
                      rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax3d.plot_surface(X, Z, Y_surf_neg, facecolors=facecolors_bottom,
                      rstride=1, cstride=1, linewidth=0, antialiased=False)

    ax3d.set_title(f"Frame {frame+1}/{len(main_df['upper_band'])}")

    vertical_line.set_xdata([frame])

ani = FuncAnimation(fig, update, frames=len(main_df["upper_band"]), repeat=True, interval=33)
#ani.save("occupation.gif", writer="pillow", fps=15)

plt.show()
