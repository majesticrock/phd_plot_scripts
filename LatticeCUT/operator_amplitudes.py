import numpy as np
import matplotlib.pyplot as plt
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *
from mrock_centralized_scripts.create_figure import *
import mrock_centralized_scripts.FullDiagPurger as fdp

N = 16000
E_F = 0
epsilon = np.linspace(-1, 1, N) - E_F

colors = [
    "#000000",
    "#F97316",
    "#5105D3",
    "#5DDFFF",
]

fig, axes_2d = create_large_figure(ncols=3, nrows=2, sharey="row", sharex="row", height_to_width_ratio=3.6/6.4)
fig.subplots_adjust(wspace=0) 
for ax in axes_2d.ravel():
    ax.set_xlabel(r"$\varepsilon / W$")

epsilon = np.linspace(-1, 1, N) - E_F

for axes, G in zip(axes_2d, [0.3, 3.0]):
    axes[0].set_ylabel(f"$g={G}$\n$\\alpha_j^{{(n)}}$")
    for ax, SYSTEM in zip(axes, ["sc", "bcc", "fcc"]):
        if G==0.3:
            ax.set_title(SYSTEM)
        params = lattice_cut_params(N=N, 
                                    g=G, 
                                    U=0.0, 
                                    E_F=E_F,
                                    omega_D=0.02)
        main_df = load_panda("lattice_cut", f"./{SYSTEM}", "full_diagonalization.json.gz", **params, print_date=False)
        purger = fdp.FullDiagPurger(main_df, epsilon)

        gap_df = load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz", **params, print_date=False)
        Delta = gap_df["Delta"]

        for PICK in range(min(len(purger.amplitude_eigenvalues), 4)):
            alpha = purger.amplitude_eigenvectors[PICK][:N]
            norm = np.max(np.abs(alpha))
            ax.plot(epsilon, alpha / norm, color=colors[PICK], label=f"{PICK+1}")

            nu = purger.amplitude_eigenvectors[PICK][N:]
            anderson = -nu * epsilon / np.where(Delta != 0, Delta, np.inf)
            ax.plot(epsilon, anderson / norm, dashes=[3.5, 3.5], c="#009100", label=r"$-\nu \varepsilon / \Delta$" if PICK+1 == min(len(purger.amplitude_eigenvalues), 4) else None)

axes_2d[-1,-1].legend(loc="lower right")
axes_2d[0,0].set_xlim(-0.05, 0.05)
axes_2d[-1,0].set_xlim(-0.25, 0.25)

plt.show()