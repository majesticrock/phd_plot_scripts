import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from get_data import load_panda, hubbard_params
import continued_fraction_pandas as cf
import plot_settings as ps
import sys
import os

def load_data(i):
    """ Returns the data for the Hubbard test with index i.
    0: SC-CDW phase
    1: SC phase
    2: CDW phase
    3: AFM phase
    """
    
    __CUBE_DIR__ = os.path.join("hubbard", "cube")
    __SQUARE_DIR__ = os.path.join("hubbard", "square")
    if i == 0:
        return load_panda(__SQUARE_DIR__, "test", "resolvents.json.gz", **hubbard_params(0.0, -2.5, 0.))
    elif i == 1:
        return load_panda(__CUBE_DIR__, "test", "resolvents.json.gz", **hubbard_params(0.0, -2.5, -0.1))
    elif i == 2:
        return load_panda(__SQUARE_DIR__, "test", "resolvents.json.gz", **hubbard_params(0.0, -2.5, 0.1))
    elif i == 3:
        return load_panda(__SQUARE_DIR__, "test", "resolvents.json.gz", **hubbard_params(0.0, 2.5, 0.1))
    else:
        raise ValueError("Hubbard test: Invalid index")

def create_plot(i):
    pd_data = load_data(i)
    resolvents = cf.ContinuedFraction(pd_data)

    fig, ax = plt.subplots()
    ax.set_ylim(-0.05, 1.)
    ax.set_xlabel(r"$\omega [t]$")
    ax.set_ylabel(r"$\mathcal{A} (\omega) [t^{-1}]$")

    plotter = ps.CURVEFAMILY(6, axis=ax)
    plotter.set_individual_colors("nice")
    plotter.set_individual_linestyles(["-", "-.", "--", "-", "--", ":"])

    w_lin = np.linspace(-0.01, pd_data["continuum_boundaries"][1] + 0.3, 5000, dtype=complex)
    w_lin += 1e-6j

    plotter.plot(w_lin, resolvents.spectral_density(w_lin, "phase_SC"), label="Phase")
    plotter.plot(w_lin, resolvents.spectral_density(w_lin, "amplitude_SC"), label="Higgs")
    plotter.plot(w_lin, resolvents.spectral_density(w_lin, "amplitude_CDW"), label="CDW")
    plotter.plot(w_lin, resolvents.spectral_density(w_lin, "amplitude_AFM"), label="l.AFM")
    plotter.plot(w_lin, resolvents.spectral_density(w_lin, "amplitude_AFM_transversal"), label="t.AFM")

    resolvents.mark_continuum(ax)

    ax.legend()
    fig.tight_layout()
    plt.show()
    
if len(sys.argv) > 1:
    create_plot(int(sys.argv[1]))
else:
    for i in range(4):
        create_plot(i)