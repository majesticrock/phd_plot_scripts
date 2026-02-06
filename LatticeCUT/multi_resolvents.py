import numpy as np
import matplotlib.pyplot as plt
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from create_zoom import *
from get_data import *
import continued_fraction_pandas as cf
import os

figs = []
axes = []
systems = ["bcc", "fcc", "sc"]

Gs = [0.5, 1, 2]
Ef = -0.5

for system in systems: 
    fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(12.8, 4.8))
    fig.subplots_adjust(wspace=0)
    figs.append(fig)
    axes.append(ax)
    
    for i, g in enumerate(Gs):
        main_df = load_panda("lattice_cut", f"test/{system}", "resolvents.json.gz",
                            **lattice_cut_params(N=2000, 
                                                 g=g, 
                                                 U=0, 
                                                 E_F=Ef,
                                                 omega_D=0.02))
        resolvents = cf.ContinuedFraction(main_df, ignore_first=5, ignore_last=60)
        
        w_lin = np.linspace(-0.005 * main_df["continuum_boundaries"][1], 0.45 * main_df["continuum_boundaries"][1], 15000, dtype=complex)
        w_lin += 1e-4j
        
        ax[i].plot(w_lin.real, resolvents.spectral_density(w_lin, "phase_SC",     withTerminator=True), label=f"Phase $g={g}$", c=f"C{i}", ls="-")
        ax[i].plot(w_lin.real, resolvents.spectral_density(w_lin, "amplitude_SC", withTerminator=True), label=f"Higgs $g={g}$", c=f"C{i}", ls=":")

        resolvents.mark_continuum(ax[i], label=None)
        ax[i].set_ylim(-0.05, 1)
        ax[i].set_xlim(np.min(w_lin.real), np.max(w_lin.real))
        ax[i].set_xlabel(r"$\omega$")
        ax[i].legend(loc="upper right")
        
    ax[1].set_title(system)
    ax[0].set_ylabel(r"$\mathcal{A} (\omega)$")
    
    #fig.savefig(f"phd_plot_scripts/LatticeCUT/build/{os.path.basename(__file__).split('.')[0]}_{system}.svg")
    
plt.show()