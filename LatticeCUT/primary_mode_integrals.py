import numpy as np
import matplotlib.pyplot as plt
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *
from mrock_centralized_scripts.create_figure import *
import mrock_centralized_scripts.FullDiagPurger as fdp

PICK = 0
N = 16000
E_F = 0

epsilon = np.linspace(-1, 1, N)
fig, ax = plt.subplots()
Gs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
      1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.8, 2.0, 2.2, 2.3, 2.5, 2.7, 2.8, 3.0]

occupation_integrals = np.zeros((3, len(Gs)))
pair_creation_integrals = np.zeros((3, len(Gs)))
systems = ['sc', 'bcc', 'fcc']

for j, system in enumerate(systems):
    for i, g in enumerate(Gs):
        params = lattice_cut_params(N=N, 
                                    g=g, 
                                    U=0.0, 
                                    E_F=E_F,
                                    omega_D=0.02)
        main_df = load_panda("lattice_cut", f"./{system}", "full_diagonalization.json.gz", print_date=False, **params)
        purger = fdp.FullDiagPurger(main_df, epsilon)
        pair_creation_integrals[j, i], occupation_integrals[j, i] = purger.integral_amplitude(0)

markers = ["o", "s", "D"]
for i in range(3):
    norms = pair_creation_integrals[i] + occupation_integrals[i]
    ax.plot(Gs, pair_creation_integrals[i] / norms, c=f"C{i}", marker=markers[i], markevery=(0. if i !=1 else 0.1, 0.2), markersize=8)
    ax.plot(Gs, occupation_integrals[i]    / norms, c=f"C{i}", marker=markers[i], markevery=(0. if i !=1 else 0.1, 0.2), markersize=8, ls="--")
    
ax.set_ylim(0, 1)
ax.set_xlabel("$g$")
ax.set_ylabel("Total contribution")
from color_and_linestyle_legends import color_and_linestyle_legends
color_and_linestyle_legends(ax, color_labels=systems, 
                            linestyle_labels=[r"$\sum_j |\alpha_j^{(1)}|^2$", r"$\sum_j |\nu_j^{(1)}|^2$"], 
                            linestyle_legend_loc="lower right")


plt.show()