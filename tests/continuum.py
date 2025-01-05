import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from get_data import load_panda, continuum_params
import continued_fraction_pandas as cf
import plot_settings as ps
import sys

def load_data(i):
    """ Returns the data for the Hubbard test with index i.
    0: No Coulomb interaction
    1: Normal screening
    2: Weak screening
    3: Strong attraction
    """
    
    if i == 0:
        return load_panda("continuum", "test", "resolvents.json.gz",
                    **continuum_params(N_k=4000, T=0, coulomb_scaling=0, screening=0, k_F=4.25, g=0.5, omega_D=10))
    elif i == 1:
        return load_panda("continuum", "test", "resolvents.json.gz",
                    **continuum_params(N_k=4000, T=0, coulomb_scaling=1, screening=1, k_F=4.25, g=0.8, omega_D=10))
    elif i == 2:
        return load_panda("continuum", "test", "resolvents.json.gz",
                    **continuum_params(N_k=4000, T=0, coulomb_scaling=1, screening=1e-4, k_F=4.25, g=1, omega_D=10))
    elif i == 3:
        return load_panda("continuum", "test", "resolvents.json.gz",
                    **continuum_params(N_k=4000, T=0, coulomb_scaling=0, screening=0, k_F=4.25, g=1.5, omega_D=10))
    else:
        raise ValueError("Continuum test: Invalid index")

def create_plot(i):
    pd_data = load_data(i)
    resolvents = cf.ContinuedFraction(pd_data, ignore_first=30, ignore_last=90)

    fig, ax = plt.subplots()
    ax.set_ylim(-0.05, 1.)
    ax.set_xlabel(r"$\omega [\mathrm{meV}]$")
    ax.set_ylabel(r"$\mathcal{A} (\omega) [\mathrm{eV}^{-1}]$")

    plotter = ps.CURVEFAMILY(6, axis=ax)
    plotter.set_individual_colors("nice")
    plotter.set_individual_linestyles(["-", "-.", "--", "-", "--", ":"])

    w_lin = np.linspace(-0.005 * pd_data["continuum_boundaries"][1], 1.1 * pd_data["continuum_boundaries"][1], 5000, dtype=complex)
    w_lin += 1e-4j

    plotter.plot(1e3 * w_lin.real, resolvents.spectral_density(w_lin, "phase_SC",     withTerminator=True), label="Phase")
    plotter.plot(1e3 * w_lin.real, resolvents.spectral_density(w_lin, "amplitude_SC", withTerminator=True), label="Higgs")

    resolvents.mark_continuum(ax, 1e3)

    ax.set_xlim(1e3 * np.min(w_lin.real), 1e3 * np.max(w_lin.real))
    ax.legend()
    fig.tight_layout()
    plt.show()
    
if len(sys.argv) > 1:
    create_plot(int(sys.argv[1]))
else:
    for i in range(4):
        create_plot(i)