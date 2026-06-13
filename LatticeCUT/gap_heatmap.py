import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *
from mrock_centralized_scripts.create_figure import *

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

E_F=-0.5
U=0.0

E_Fs = [-0.5, -0.3, -0.2, -0.1]
Us = [0,  0.01, 0.05, 0.1, 0.5]

sweep_ef = False

fig, axes = plt.subplots(ncols=len(E_Fs) if sweep_ef else len(Us), figsize=(12, 6))
main_df = load_pickle(f"lattice_cut/bcc/N=16000/", "gaps.pkl")

for ax, sweep in zip(axes, (E_Fs if sweep_ef else Us)):
    if sweep_ef:
        E_F = sweep
    else:
        U = sweep
    queried = main_df.query(
        f"omega_D == 0.02 & E_F == {E_F} & U=={U}"
    ).sort_values("g")
    energies = queried["energies"].iloc[0]
    gs = queried["g"].to_numpy()

    X, Y = np.meshgrid(energies, gs)
    Z = np.stack(queried["Delta"].to_numpy())
    for i in range(len(gs)):
        if Z[i,-1] > 0.0:
            Z[i,:] *= -1

    _max = np.max(np.abs(Z))
    norm = TwoSlopeNorm(vmin=-_max, vcenter=0.0, vmax=_max)

    contour = ax.contourf(X, Y, Z, cmap=plt.get_cmap("seismic"), levels=255, zorder=-20, extend="max", norm=norm)
    ax.plot(queried["chemical_potential"], queried["g"], ls="--", c="k", alpha=0.6)
    ax.set_rasterization_zorder(-10)
    ax.set_xlabel(r"$\varepsilon / W$")

    cbar = fig.colorbar(contour, ax=ax, orientation='horizontal')
cbar.set_label(r"$|\Delta(\varepsilon)| / W$")

axes[0].set_ylabel("$g$")

plt.show()