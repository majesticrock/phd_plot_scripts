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

systems = ['sc', 'bcc', 'fcc']
markers = ["o", "s", "D"]

for j, system in enumerate(systems):
    main_df = load_pickle(f"lattice_cut/{system}/N={N}", "full_diagonalizations.pkl").query(
        f"E_F=={E_F} & omega_D==0.02 & U==0"
    ).sort_values("g").reset_index(drop=True)
    Gs = main_df["g"]
    occupation_integrals = np.zeros(len(Gs))
    pair_creation_integrals = np.zeros(len(Gs))
    
    for i, row in main_df.iterrows():
        purger = fdp.FullDiagPurger(row, epsilon)
        if len(purger.amplitude_eigenvalues) > 1:
            pair_creation_integrals[i], occupation_integrals[i] = purger.integral_amplitude(1)

    norms = pair_creation_integrals + occupation_integrals
    ax.plot(Gs, pair_creation_integrals / np.where(norms > 0, norms, np.nan), c=f"C{j}")
    ax.plot(Gs, occupation_integrals    / np.where(norms > 0, norms, np.nan), c=f"C{j}", ls="--")
    
ax.set_ylim(0, 1)
ax.set_xlabel("$g$")
ax.set_ylabel("Total contribution")
from color_and_linestyle_legends import color_and_linestyle_legends
color_and_linestyle_legends(ax, color_labels=systems, 
                            linestyle_labels=[r"$\sum_j |\alpha_j^{(1)}|^2$", r"$\sum_j |\nu_j^{(1)}|^2$"], 
                            linestyle_legend_loc="lower right")


plt.show()