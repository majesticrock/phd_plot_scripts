import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

def create_frame(nrows=1, ylabel_list=None, **kwargs):
    fig, ax = plt.subplots(nrows=nrows, sharex=True, **kwargs)
    if (nrows == 1):
        if ylabel_list is None:
            ax.set_xlabel(legend(r"t / T_\mathrm{L}"))
        else:
            ax.set_xlabel(ylabel_list)
        ax.set_ylabel(legend(r"j(t)", "normalized"))
    else:
        ax[-1].set_xlabel(legend(r"t / T_\mathrm{L}"))
        for i, a in enumerate(ax):
            if ylabel_list is None:
                a.set_ylabel(legend(r"j(t)", "normalized"))
            else:
                a.set_ylabel(ylabel_list[i])
    fig.tight_layout()
    return fig, ax

def add_current_density_to_plot(df, ax, label=None, sigma=None, normalize=True, substract=None, **kwargs):
    times = np.linspace(0, df["t_end"] - df["t_begin"], len(df["current_density_time"])) / (2 * np.pi)
    if substract is None:
        normalization = 1. / np.max(np.abs(df["current_density_time"])) if normalize else 1.0
        y_data = df["current_density_time"] * normalization
    else:
        y_data = df["current_density_time"] - substract(times)
        if normalize:
            y_data /= np.max(y_data)
    ax.plot(times, y_data, label=label, **kwargs)
    if sigma is not None:
        worst_case = 4 * sigma * normalization
        ax.fill_between(times, y_data - worst_case, y_data + worst_case, alpha=0.2, **kwargs)

def plot_j(df):
    fig, ax = create_frame()
    add_current_density_to_plot(df, ax)
    ax.legend()

    plt.show()

if __name__ == '__main__':
    fig, ax = create_frame()
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