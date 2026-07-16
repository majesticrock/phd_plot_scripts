import numpy as np
import matplotlib.pyplot as plt
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import load_panda, lattice_cut_params
import continued_fraction_pandas as cf
import plot_settings as ps
import sys

def plot_name(i):
    if i == 0:
        return "SC $g=2.5$"
    elif i == 1:
        return "$BCC $g=1.5$ $U=0.2$"
    
def load_data(i):
    """ Returns the data for the Hubbard test with index i.
    0: No Coulomb interaction
    1: Normal screening
    2: Weak screening
    3: Strong attraction
    """
    
    if i == 0:
        return load_panda("lattice_cut", "test", "full_diagonalization.json.gz",
                    **lattice_cut_params(N=2000, g=1., U=0., E_F=0, omega_D=0.02))
    elif i == 1:
        return load_panda("lattice_cut", "test", "full_diagonalization.json.gz",
                    **lattice_cut_params(N=2000, g=1., U=0., E_F=0, omega_D=0.02))
    else:
        raise ValueError("Continuum test: Invalid index")

def create_plot(i):
    fig, ax = plt.subplots()
    #TODO
    plt.show()


if len(sys.argv) > 1:
    create_plot(int(sys.argv[1]))
else:
    for i in range(4):
        create_plot(i)