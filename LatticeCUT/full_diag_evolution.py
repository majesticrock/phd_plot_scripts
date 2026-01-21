import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from get_data import *

PICK = 1
SYSTEM = 'bcc'
N=16000
E_F = 0

fig_wv, axes_wv = plt.subplots(nrows=3, sharex=True)
fig_wv.subplots_adjust(hspace=0)
axes_wv[0].set_ylabel(r"$c_{k\uparrow}^\dagger c_{-k\downarrow}^\dagger + \mathrm{H.c}$")
axes_wv[1].set_ylabel(r"$c_{k\uparrow}^\dagger c_{k\uparrow} + c_{k\downarrow}^\dagger c_{k\downarrow}$")
axes_wv[2].set_ylabel(r"$c_{k\uparrow}^\dagger c_{-k\downarrow}^\dagger - \mathrm{H.c}$")
axes_wv[-1].set_xlabel(r"$\varepsilon - E_\mathrm{F}$")
epsilon = np.linspace(-1, 1, N) - E_F

def add_line(ax, y, **kwargs):
    y = np.asarray(y)
    if len(y) != N and len(y) != 2*N:
        return
    if np.sum(y) < 0:#y[0] < 0:
            y = -y
            
    if len(y) == N:
        norm = np.max(np.abs(y))
        ax.plot(epsilon, y / norm, **kwargs)
    elif len(y) == 2*N:
        norm = np.max(np.abs(y[:N]))
        ax[0].plot(epsilon, y[:N] / norm, **kwargs)
        norm = np.max(np.abs(y[N:]))
        ax[1].plot(epsilon, y[N:] / norm, **kwargs)

#Gs = [2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.7, 2.65, 2.8, 2.9, 3.0]
Gs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
cmap = plt.get_cmap('plasma', len(Gs))
for i, g in enumerate(Gs):
    params = lattice_cut_params(N=N, 
                                g=g, 
                                U=0.0, 
                                E_F=E_F,
                                omega_D=0.02)
    main_df = load_panda("lattice_cut", f"./{SYSTEM}", "full_diagonalization.json.gz", **params)

    add_line(axes_wv[:2], main_df["amplitude.first_eigenvectors"][PICK], label=f"$g={g}$", c=cmap(i / len(Gs)))
    add_line(axes_wv[2],  main_df["phase.first_eigenvectors"][PICK-1], label=f"$g={g}$", c=cmap(i / len(Gs)))

axes_wv[0].legend(loc="upper right")
axes_wv[0].set_xlim(-0.2, 0.2)

plt.show()