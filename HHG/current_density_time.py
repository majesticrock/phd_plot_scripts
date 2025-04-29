import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

def plot_data(df, ax, label=None, sigma=None, **kwargs):
    times = np.linspace(df["t_begin"], df["t_end"], len(df["current_density_time"])) / (2 * np.pi)
    normalization = 1. / np.max(np.abs(df["current_density_time"]))
    y_data = df["current_density_time"] * normalization
    ax.plot(times, y_data, label=label, **kwargs)
    if sigma is not None:
        worst_case = 4 * sigma * normalization
        ax.fill_between(times, y_data - worst_case, y_data + worst_case, alpha=0.2, **kwargs)

def plot_j(df):
    fig, ax = plt.subplots()
    plot_data(df, ax)
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

    #main_df = load_panda("HHG", "test_200/cosine_laser/PiFlux", "current_density.json.gz", 
    #                 **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    #plot_data(main_df, ax, "200")
    #main_df = load_panda("HHG", "test_400/cosine_laser/PiFlux", "current_density.json.gz", 
    #                 **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    #plot_data(main_df, ax, "400")
    #main_df = load_panda("HHG", "test_cont_asym_30/cosine_laser/PiFlux", "current_density.json.gz", 
    #                     **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    #plot_data(main_df, ax, "G. 240/30")
    #main_df = load_panda("HHG", "test_cont_asym_120/cosine_laser/PiFlux", "current_density.json.gz", 
    #                     **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    #plot_data(main_df, ax, "G. 240/120")
    #main_df = load_panda("HHG", "test_cont_asym_120_120/cosine_laser/PiFlux", "current_density.json.gz", 
    #                     **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    #plot_data(main_df, ax, "G. 120/120")
    main_df = load_panda("HHG", "test_cont_240/cosine_laser/PiFlux", "current_density.json.gz", 
                         **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    plot_data(main_df, ax, "G. 240/240", ls="-", color="C0")
    main_df = load_panda("HHG", "test_mc/cosine_laser/PiFlux", "current_density.json.gz", 
                         **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    plot_data(main_df, ax, "MC", ls="-.", color="C1", sigma=1./np.sqrt(1e7))
    main_df = load_panda("HHG", "test_mc_2/cosine_laser/PiFlux", "current_density.json.gz", 
                         **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    plot_data(main_df, ax, "MC2", ls="-.", color="C2", sigma=1./np.sqrt(64e6))
    #main_df = load_panda("HHG", "test_mc_3/cosine_laser/PiFlux", "current_density.json.gz", 
    #                     **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    #plot_data(main_df, ax, "MC3", ls="--", color="C3", sigma=1./np.sqrt(1e6))
    #main_df = load_panda("HHG", "test_mc_4/cosine_laser/PiFlux", "current_density.json.gz", 
    #                     **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    #plot_data(main_df, ax, "MC4", ls="--", color="C4", sigma=1./np.sqrt(1e7))
    main_df = load_panda("HHG", "test/cosine_laser/PiFlux", "current_density.json.gz", 
                         **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=10))
    plot_data(main_df, ax, "Decay", ls="--", color="C4")
    
    ax.legend()
    plt.show()