import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from get_data import *

PICK = 4
SYSTEM = 'fcc'
N=16000

fig_wv, axes_wv = plt.subplots(nrows=3, sharex=True, sharey=True)
fig_wv.subplots_adjust(hspace=0)
axes_wv[0].set_ylabel("Higgs")
axes_wv[1].set_ylabel("Occupation")
axes_wv[2].set_ylabel("Phase")
axes_wv[-1].set_xlabel(r"$\varepsilon$")
epsilon = np.linspace(-1, 1, N)

def add_line(ax, y, **kwargs):
    y = np.asarray(y)
    if len(y) != N and len(y) != 2*N:
        return
    if y[0] < 0:
            y = -y
            
    norm = np.max(np.abs(y))
    if len(y) == N:
        ax.plot(epsilon, y / norm, **kwargs)
    elif len(y) == 2*N:
        ax[0].plot(epsilon, y[:N] / norm, **kwargs)
        ax[1].plot(epsilon, y[N:] / norm, **kwargs)

Gs = [2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.8, 2.9, 3.0]
cmap = plt.get_cmap('plasma', len(Gs))
for i, g in enumerate(Gs):
    params = lattice_cut_params(N=N, 
                                g=g, 
                                U=0.0, 
                                E_F=-0.5,
                                omega_D=0.02)
    main_df = load_panda("lattice_cut", f"./{SYSTEM}", "full_diagonalization.json.gz", **params)

    add_line(axes_wv[:2], main_df["amplitude.first_eigenvectors"][PICK], label=f"$g={g}$", c=cmap(i / len(Gs)))
    add_line(axes_wv[2], main_df["phase.first_eigenvectors"][PICK], label=f"$g={g}$", c=cmap(i / len(Gs)))

axes_wv[0].legend(loc="upper right")

plt.show()