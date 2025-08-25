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

def add_laser_to_plot(df, ax, electric_field=False, label=None, normalize=False, **kwargs):
    times = np.linspace(0, df["t_end"] - df["t_begin"], df["n_measurements"]) / (2 * np.pi)
    A = df["laser_function"]

    if electric_field:
        dt = (times[1] - times[0])
        E = -np.gradient(A, dt)
        if normalize:
            E /= np.max(np.abs(E))
        ax.plot(times, E, label=label, **kwargs)
    else:
        if normalize:
            ax.plot(times, A / np.max(np.abs(A)), label=label, **kwargs)
        else:
            ax.plot(times, A, label=label, **kwargs)

def plot_laser(df, electric_field=False):
    fig, ax = create_frame(electric_field)
    add_laser_to_plot(df, ax, electric_field)
    
    ax.axhline(0, c="k", ls="--", linewidth=1.5)
    plt.show()

if __name__ == '__main__':
    __PLOT_E_FIELD__ = False
    fig, ax = create_frame(__PLOT_E_FIELD__)

    main_df = load_panda("HHG", "cascade_prec/exp_laser/PiFlux", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=1.5e6, band_width=200, field_amplitude=1., photon_energy=1, tau_diag=10, tau_offdiag=-1, t0=0))
    #add_laser_to_plot(main_df, ax, False, label="Vector potential")
    add_laser_to_plot(main_df, ax, True, label="Electric field", normalize=True)
    ax.axhline(0, c="k", ls="--", linewidth=1.5)
    
    N_extra = 16
    time_scale = 6.67111 * main_df["photon_energy"] / (6.582119569509065698e-1 * 2 * np.pi)
    dt = time_scale / 201 # number of sample points of the electric field
    ax.axvline(N_extra * dt, c="k")
    ax.axvline(time_scale + N_extra * dt, c="k")
    
    import os
    EXP_PATH = "../raw_data_phd/" if os.name == "nt" else "data/"
    LASER = np.loadtxt(f"{EXP_PATH}HHG/pulse_AB.dat").transpose()
    
    ax.plot((16 * 0.03318960199004975 + LASER[0]) * main_df["photon_energy"] / (6.582119569509065698e-1 * 2 * np.pi), (LASER[1] + LASER[2]) / np.max(np.abs(LASER[1] + LASER[2])), label="Exp", ls="--")
    
    plt.show()