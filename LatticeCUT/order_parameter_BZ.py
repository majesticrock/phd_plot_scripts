import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

N = 200
K_Z = 0.
SYSTEM = 'bcc'

def dispersion(x, y, z):
    if SYSTEM == 'sc':
        return (np.cos(np.pi * x) + np.cos(np.pi * y) + np.cos(np.pi * z)) / 3.
    elif SYSTEM == 'bcc':
        return (np.cos(0.5 * np.pi * x) * np.cos(0.5 * np.pi * y) * np.cos(0.5 * np.pi * z))
    elif SYSTEM == 'fcc':
        return 0.5 - 0.5 * (np.cos(0.5 * np.pi * x) * np.cos(0.5 * np.pi * y) + np.cos(0.5 * np.pi * z) * np.cos(0.5 * np.pi * y) + np.cos(0.5 * np.pi * x) * np.cos(0.5 * np.pi * z))

base_df = load_panda('lattice_cut', f'./{SYSTEM}', 'gap.json.gz',
                    **lattice_cut_params(N=16000, 
                                         g=1.5,
                                         U=0, 
                                         E_F=-0.5,
                                         omega_D=0.02))
main_df = load_panda('lattice_cut', f'./{SYSTEM}', 'gap.json.gz',
                    **lattice_cut_params(N=16000, 
                                         g=2,
                                         U=0, 
                                         E_F=-0.5,
                                         omega_D=0.02))
from scipy.interpolate import interp1d
base_gaps = interp1d(base_df['energies'], base_df['Delta'], assume_sorted=True, fill_value='extrapolate', bounds_error=False)
enh_gaps  = interp1d(main_df['energies'], main_df['Delta'], assume_sorted=True, fill_value='extrapolate', bounds_error=False)

X, Y = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))

fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6.4, 9.6))
Z = enh_gaps(dispersion(X, Y, K_Z))
levels = np.linspace(np.min(main_df["Delta"]), np.max(main_df["Delta"]), 61)
cont = axes[1].contourf(X, Y, Z, cmap="viridis", levels=levels)

Z = base_gaps(dispersion(X, Y, K_Z))
axes[0].contourf(X, Y, Z, cmap="viridis", levels=levels)

cbar = fig.colorbar(cont, ax=axes)
cbar.set_label("$\\Delta / W$")
axes[0].set_xlabel("$k_x / \\pi$")
axes[0].set_ylabel("$g=1.5$\n$k_y / \\pi$")
axes[1].set_ylabel("$g=2$\n$k_y / \\pi$")


############# high symmetry
def qp_dispersion(epsilons, gaps):
    return np.sqrt(epsilons**2 + gaps(epsilons)**2)


GAMMA = np.array([0, 0, 0])
H     = np.array([0, 0, 2*np.pi])
N     = np.array([np.pi, np.pi, 0])
P     = np.array([np.pi, np.pi, np.pi])

N_Q = 500
q = np.linspace(0, 1, N_Q)

def high_symmetry_line(begin, end, gaps, ax, ax_shift, **kwargs):
    q_line = begin + (end - begin) * q[:,None]
    epsilons = dispersion(q_line[:,0], q_line[:,1], q_line[:,2])
    
    #ax.plot(q + ax_shift, gaps(epsilons), **kwargs)
    ax.plot(q + ax_shift, qp_dispersion(epsilons, gaps), **kwargs)
    ax.axvline(ax_shift, c="k")

fig_hs, ax_hs = plt.subplots()
high_symmetry_line(GAMMA, N, enh_gaps, ax_hs, 0, c="C0", label="$g=2$")
high_symmetry_line(N, H,     enh_gaps, ax_hs, 1, c="C0")
high_symmetry_line(H, GAMMA, enh_gaps, ax_hs, 2, c="C0")
high_symmetry_line(GAMMA, P, enh_gaps, ax_hs, 3, c="C0")

ax_hs.set_xticks([0,1,2,3,4], [r"$\Gamma$", r"$N$", r"$H$", r"$\Gamma$", r"$P$"])
ax_hs.set_ylabel(r"$E / W$")
ax_hs.set_xlabel(r"$k$")




main_df = load_panda('lattice_cut', f'./{SYSTEM}', 'gap.json.gz',
                    **lattice_cut_params(N=16000, 
                                         g=1.5,
                                         U=0, 
                                         E_F=-0.5,
                                         omega_D=0.02))
from scipy.interpolate import interp1d
gaps = interp1d(main_df['energies'], main_df['Delta'], assume_sorted=True, fill_value='extrapolate', bounds_error=False)
high_symmetry_line(GAMMA, N, base_gaps, ax_hs, 0, c="C1", ls="--", label="$g=1.5$")
high_symmetry_line(N, H,     base_gaps, ax_hs, 1, c="C1", ls="--")
high_symmetry_line(H, GAMMA, base_gaps, ax_hs, 2, c="C1", ls="--")
high_symmetry_line(GAMMA, P, base_gaps, ax_hs, 3, c="C1", ls="--")

ax_hs.legend(loc="upper right")

plt.show()