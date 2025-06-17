import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import legend

def create_frame(electric_field=False):
    fig, ax = plt.subplots()
    ax.set_xlabel(legend(r"t / (2 \pi T_L)"))
    ax.set_ylabel(f"${'E(t)' if electric_field else 'A(t)'} $ arb. units")
    fig.tight_layout()
    
    return fig, ax

def add_laser_to_plot(df, ax, electric_field=False, label=None, **kwargs):
    times = np.linspace(0, df["t_end"] - df["t_begin"], df["n_measurements"]) / (2 * np.pi)
    A = df["laser_function"]

    if electric_field:
        dt = (times[1] - times[0])
        E = -np.gradient(A, dt)
        ax.plot(times, E, label=label, **kwargs)
    else:
        ax.plot(times, A, label=label, **kwargs)

def plot_laser(df, electric_field=False):
    fig, ax = create_frame(electric_field)
    add_laser_to_plot(df, ax, electric_field)
    
    ax.axhline(0, c="k", ls="--", linewidth=1.5)
    plt.show()

if __name__ == '__main__':
    __PLOT_E_FIELD__ = True
    fig, ax = create_frame(__PLOT_E_FIELD__)
    
    main_df = load_panda("HHG", "test/expA_laser/PiFlux", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1., photon_energy=1, decay_time=100))
    add_laser_to_plot(main_df, ax, __PLOT_E_FIELD__)

    main_df = load_panda("HHG", "test/expB_laser/PiFlux", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=1.5e5, band_width=400, field_amplitude=1., photon_energy=1, decay_time=100))
    add_laser_to_plot(main_df, ax, __PLOT_E_FIELD__)
    ax.axhline(0, c="k", ls="--", linewidth=1.5)
    
    dt = main_df["t_end"] / main_df["n_measurements"]
    ax.axvline(16 * dt, c="k")
    ax.axvline()
    plt.show()