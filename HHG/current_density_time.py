import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

def add_current_density_to_plot(df, ax, label=None, sigma=None, **kwargs):
    times = np.linspace(0, df["t_end"] - df["t_begin"], len(df["current_density_time"])) / (2 * np.pi)
    normalization = 1. / np.max(np.abs(df["current_density_time"]))
    y_data = df["current_density_time"] * normalization
    ax.plot(times, y_data, label=label, **kwargs)
    if sigma is not None:
        worst_case = 4 * sigma * normalization
        ax.fill_between(times, y_data - worst_case, y_data + worst_case, alpha=0.2, **kwargs)

def plot_j(df):
    fig, ax = plt.subplots()
    add_current_density_to_plot(df, ax)
    ax.set_xlabel(legend(r"t / T_\mathrm{L}")) # T_L = 2 pi / omega_L
    ax.set_ylabel(legend(r"j(t)"))
    ax.legend()

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.set_xlabel(legend(r"t / T_\mathrm{L}")) # T_L = 2 pi / omega_L
    ax.set_ylabel(legend(r"j(t)", "normalized"))

    fig.tight_layout()

    main_df = load_panda("HHG", "test0/cosine_laser/PiFlux", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    add_current_density_to_plot(main_df, ax, f"$0$")
    main_df = load_panda("HHG", "test1/cosine_laser/PiFlux", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    add_current_density_to_plot(main_df, ax, f"$1$", ls="--")
    main_df = load_panda("HHG", "test05/cosine_laser/PiFlux", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    add_current_density_to_plot(main_df, ax, f"$0.5$", ls="-.")
    
    ax.legend()
    plt.show()