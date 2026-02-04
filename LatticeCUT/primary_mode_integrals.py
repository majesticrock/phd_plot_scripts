import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from get_data import *
from create_figure import *
import FullDiagPurger as fdp

PICK = 0
N = 16000
E_F = 0

epsilon = np.linspace(-1, 1, N)
fig, ax = plt.subplots()
Gs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0]

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

for i in range(3):
    norms = pair_creation_integrals[i] + occupation_integrals[i]
    ax.plot(Gs, pair_creation_integrals[i] / norms, c=f"C{i}")
    ax.plot(Gs, occupation_integrals[i]    / norms, c=f"C{i}", ls="--")

ax.set_xlabel("$g$")
ax.set_ylabel("Total contribution")
from color_and_linestyle_legends import color_and_linestyle_legends
color_and_linestyle_legends(ax, color_labels=systems, linestyle_labels=[r"$\sum_j |\alpha_j^{(1)}|^2$", r"$\sum_j |\nu_j^{(1)}|^2$"], linestyle_legend_loc="upper center")


plt.show()