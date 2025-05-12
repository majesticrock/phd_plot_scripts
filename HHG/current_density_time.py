import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

def add_current_density_to_plot(df, ax, label=None, sigma=None, **kwargs):
    times = np.linspace(df["t_begin"], df["t_end"], len(df["current_density_time"])) / (2 * np.pi)
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

    main_df = load_panda("HHG", "test/cosine_laser/PiFlux", "current_density.json.gz", 
                     **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    add_current_density_to_plot(main_df, ax, "4/60")
    main_df = load_panda("HHG", "test2/cosine_laser/PiFlux", "current_density.json.gz", 
                         **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    add_current_density_to_plot(main_df, ax, "8/60")
    main_df = load_panda("HHG", "test3/cosine_laser/PiFlux", "current_density.json.gz", 
                         **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    add_current_density_to_plot(main_df, ax, "8/120")
    main_df = load_panda("HHG", "test4/cosine_laser/PiFlux", "current_density.json.gz", 
                         **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    add_current_density_to_plot(main_df, ax, "16/60")
    main_df = load_panda("HHG", "test5/cosine_laser/PiFlux", "current_density.json.gz", 
                         **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    add_current_density_to_plot(main_df, ax, "16/120")
    main_df = load_panda("HHG", "test7/cosine_laser/PiFlux", "current_density.json.gz", 
                         **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    add_current_density_to_plot(main_df, ax, "16/240")
    
    #main_df = load_panda("HHG", "test5/cosine_laser/PiFlux", "current_density.json.gz", 
    #                 **hhg_params(T=300, E_F=118, v_F=1.5e6, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    #add_current_density_to_plot(main_df, ax, "16/120")
    #main_df = load_panda("HHG", "test6/cosine_laser/PiFlux", "current_density.json.gz", 
    #                     **hhg_params(T=300, E_F=118, v_F=1.5e6, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    #add_current_density_to_plot(main_df, ax, "32/120")
    #main_df = load_panda("HHG", "test7/cosine_laser/PiFlux", "current_density.json.gz", 
    #                     **hhg_params(T=300, E_F=118, v_F=1.5e6, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    #add_current_density_to_plot(main_df, ax, "16/240", ls="--")
    
    ax.legend()
    plt.show()