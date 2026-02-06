import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation

# Parameters
N = 200
frames = 400

X, Y = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))
levels = np.linspace(0, 1, 21, endpoint=True)

def dispersion(x, y, z, system):
    if system == 'sc':
        return (np.cos(np.pi * x) + np.cos(np.pi * y) + np.cos(np.pi * z)) / 3.
    elif system == 'bcc':
        return (np.cos(0.5 * np.pi * x) *
                np.cos(0.5 * np.pi * y) *
                np.cos(0.5 * np.pi * z))
    elif system == 'fcc':
        return 0.5 - 0.5 * (
            np.cos(0.5 * np.pi * x) * np.cos(0.5 * np.pi * y) +
            np.cos(0.5 * np.pi * z) * np.cos(0.5 * np.pi * y) +
            np.cos(0.5 * np.pi * x) * np.cos(0.5 * np.pi * z))

EFs = [0, -0.5]
SYSTEMS = ["sc", "bcc", "fcc"]
dfs = [ [ load_panda('lattice_cut', f'./{system}', 'gap.json.gz',
                        **lattice_cut_params(N=16000,
                                             g=2.,
                                             U=0,
                                             E_F=Ef,
                                             omega_D=0.02)) for system in SYSTEMS ] for Ef in EFs ]
all_gaps = np.array([ [interp1d(df['energies'], df['Delta'] / np.max(df['Delta']), assume_sorted=True, fill_value='extrapolate', bounds_error=False) for df in _dfs ] for _dfs in dfs ])

def set_axes_labels(axes):
    axes[-1][0].set_xlabel("$k_x / \\pi$")
    axes[-1][1].set_xlabel("$k_x / \\pi$")
    axes[-1][2].set_xlabel("$k_x / \\pi$")

    axes[0][0].set_ylabel("$k_y / \\pi$")
    axes[1][0].set_ylabel("$k_y / \\pi$")

fig, axes = plt.subplots(nrows=len(EFs), ncols=len(SYSTEMS), figsize=(12, 6), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.1, hspace=0.1)

kz_values = np.linspace(-2, 2, frames, endpoint=False)
disp_arrays = np.array([
    [dispersion(X, Y, kz, system) for system in SYSTEMS]
    for kz in kz_values
])

imshows = []
for system, axs in enumerate(axes.transpose()):
    disp = disp_arrays[0][system]
    for Ef, ax in enumerate(axs):
        Z = (all_gaps[Ef][system])(disp)
        im = ax.imshow(Z, origin='lower', extent=[-2, 2, -2, 2], vmin=0, vmax=1, cmap="magma", aspect='auto')
        imshows.append(im)

cbar = fig.colorbar(imshows[0], ax=axes.ravel().tolist())
cbar.set_label("$\\Delta (k) / \\Delta_\\mathrm{max}$")
set_axes_labels(axes)
axes[0][1].set_title(f"$k_z/\\pi = {kz_values[0]:.2f}$")

def update(frame):
    kz = kz_values[frame]
    for system, axs in enumerate(axes.transpose()):
        disp = disp_arrays[frame][system]
        for Ef, ax in enumerate(axs):
            Z = (all_gaps[Ef][system])(disp)
            imshows[system*len(EFs)+Ef].set_data(Z)
    axes[0][1].set_title(f"$k_z/\\pi = {kz:.2f}$")
    return imshows

ani = FuncAnimation(fig, update, frames=frames, blit=False, interval=66, repeat=True)

ani.save("topology_animation.mp4", writer="ffmpeg", dpi=150)
#plt.show()