if __name__ != '__main__x':
    import matplotlib
    
    print("Current backend:", matplotlib.get_backend())

import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

def add_current_density_to_plot(main_df, ax, **plot_kwargs):
    frequencies = main_df["frequencies"]
    current_density = frequencies * (main_df["current_density_frequency_real"] + 1.0j * main_df["current_density_frequency_imag"])
    current_density += 1.0j * main_df["current_density_time"][-1] * np.exp(1.0j * main_df["t_end"] * frequencies)
    y_data = np.abs(current_density)
    ax.plot(frequencies, y_data / np.max(y_data), **plot_kwargs)

def add_verticals(frequencies, ax):
    for i in range(1, int(np.max(frequencies)) + 1, 2):
        ax.axvline(i, ls="--", color="grey", linewidth=1, alpha=0.5)

def create_frame():
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xlabel(legend(r"\omega / \omega_L"))
    ax.set_ylabel(legend(r"j(\omega)", "normalized"))
    
    return fig, ax

def plot_j(main_df):
    fig, ax = create_frame()
    
    frequencies = main_df["frequencies"]
    add_verticals(frequencies, ax)
    add_current_density_to_plot(main_df, ax)

    fig.tight_layout()
    plt.show()
    
    
if __name__ == '__main__':
    main_df = load_panda("HHG", "test/cosine_laser/PiFlux", "current_density.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    plot_j(main_df)