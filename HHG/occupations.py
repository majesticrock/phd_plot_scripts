import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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

for i, laser in enumerate(main_df["laser_function"]):
    print(i, ":", laser)
print("Select time:")
time = int(input())

while time >= 0:
    data_top = main_df["upper_band"][time]
    data_bottom = main_df["lower_band"][time]

    nx, nz = data_top.shape
    x = np.linspace(0, np.pi, nx)
    z = np.linspace(-np.pi, np.pi, nz)
    X, Z = np.meshgrid(x, z, indexing='ij')

    # 5.889401182228545meV = photon energy
    #Y_surf = 275 * 5.889401182228545 * np.sqrt(np.cos(X)**2 + np.cos(Z)**2)
    Y_surf = main_df["dispersion"][time]
    Y_surf_neg = -Y_surf

    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = cm.viridis

    facecolors_top = cmap(norm(data_top))
    facecolors_bottom = cmap(norm(data_bottom))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Top surface (+sqrt)
    ax.plot_surface(X, Z, Y_surf,
                    facecolors=facecolors_top,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=False)

    # Bottom surface (-sqrt)
    ax.plot_surface(X, Z, Y_surf_neg,
                    facecolors=facecolors_bottom,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=False)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.concatenate([data_top.ravel(), data_bottom.ravel()]))
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label='Occupation')

    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_z$')
    ax.set_zlabel('$E(k_x, \\pi / 2, k_z)$ (meV)')
    ax.view_init(elev=47, azim=-30)

    plt.show()
    time = int(input())