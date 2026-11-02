import numpy as np
import matplotlib.pyplot as plt


from mrock.get_data import *
data_loader = DataLoader()
from mrock_centralized_scripts.legend import  legend

def create_frame(electric_field=False):
    fig, ax = plt.subplots()
    ax.set_xlabel(legend(r"t / T_L"))
    ax.set_ylabel(f"${'E(t)' if electric_field else 'A(t)'} $ arb. units")
    fig.tight_layout()
    
    return fig, ax

def add_laser_to_plot(df, ax, electric_field=False, label=None, normalize=False, **kwargs):
    times = np.linspace(0, df["t_end"] - df["t_begin"], df["n_measurements"])
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
    HBAR = 0.6582119569509065
    __PLOT_E_FIELD__ = True
    fig, ax = create_frame(__PLOT_E_FIELD__)

    main_df = data_loader.load_panda("HHG", "cascade2/exp_laser/PiFlux", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=2.5e6, band_width=600, field_amplitude=1., photon_energy=1, tau_diag=5, tau_offdiag=-1, t0=0))
    #add_laser_to_plot(main_df, ax, False, label="Vector potential", normalize=True)
    add_laser_to_plot(main_df, ax, True,  label="Electric field", normalize=True)
    N_extra = 7.5
    
    CONVERT_TO_UNITLESS = main_df["photon_energy"] / HBAR
    T_SHIFT = N_extra * 0.03318960199004975 * CONVERT_TO_UNITLESS
    
    ax.axhline(0, ls="-", linewidth=1.5)
    
    time_scale = 6.67111 * CONVERT_TO_UNITLESS
    dt = time_scale / 201 # number of sample points of the electric field
    ax.axvline(N_extra * dt, c="k")
    ax.axvline(time_scale + N_extra * dt, c="k")
    
    import os
    EXP_PATH = "../raw_data_phd/" if os.name == "nt" else "data/"
    LASER = np.loadtxt(f"{EXP_PATH}HHG/pulse_AB.dat").transpose()
    
    ax.plot(T_SHIFT + LASER[0] * CONVERT_TO_UNITLESS, (LASER[1] + LASER[2]) / np.max(np.abs(LASER[1] + LASER[2])), 
            label="Exp", ls="--")
    ax.legend()
    plt.show()