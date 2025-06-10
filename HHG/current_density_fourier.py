if __name__ != '__main__x':
    import matplotlib
    
    print("Current backend:", matplotlib.get_backend())

import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

def compute_current_density(main_df):
    frequencies = main_df["frequencies"]
    current_density = frequencies * (main_df["current_density_frequency_real"] + 1.0j * main_df["current_density_frequency_imag"])
    if main_df["decay_time"] > 0:
        current_density += (1.0j * main_df["current_density_time"][-1] * frequencies / ((1. / main_df["decay_time"]) + 1.0j * frequencies)) * np.exp(-main_df["t_end"] * ((1. / main_df["decay_time"]) + 1.0j * frequencies))
    else:
        current_density += main_df["current_density_time"][-1] * np.exp(-1.0j * main_df["t_end"] * frequencies)
    return current_density
    
def add_current_density_to_plot(main_df, ax, label=None, shift=1, max_freq=None, substract=None, normalize=True, **plot_kwargs):
    frequencies = main_df["frequencies"]
    y_data = np.abs(compute_current_density(main_df)) if substract is None else np.abs(compute_current_density(main_df) - substract)
    
    if max_freq is not None:
        mask = frequencies < max_freq
        frequencies = frequencies[mask]
        y_data = y_data[mask]
    if normalize:
        y_data /= np.max(y_data)
    ax.plot(frequencies, shift * y_data, label=label, **plot_kwargs)

def add_verticals(frequencies, ax, max_freq=None, positions='odd'):
    shift = 1 if positions=='odd' else 0
    if max_freq is not None:
        for i in range(shift, int(np.max(frequencies[frequencies < max_freq])) + 1, 2):
            ax.axvline(i, ls="--", color="grey", linewidth=1, alpha=0.5)
    else:
        for i in range(shift, int(np.max(frequencies)) + 1, 2):
            ax.axvline(i, ls="--", color="grey", linewidth=1, alpha=0.5)

def create_frame(nrows=1, ylabel_list=None, **kwargs):
    fig, ax = plt.subplots(nrows=nrows, sharex=True, **kwargs)
    if (nrows == 1):
        ax.set_yscale("log")
        if ylabel_list is None:
            ax.set_xlabel(legend(r"\omega / \omega_\mathrm{L}"))
        else:
            ax.set_xlabel(ylabel_list)
        ax.set_ylabel(legend(r"|\omega j(\omega)|", "normalized"))
    else:
        ax[-1].set_xlabel(legend(r"\omega / \omega_\mathrm{L}"))
        for i, a in enumerate(ax):
            a.set_yscale("log")
            if ylabel_list is None:
                a.set_ylabel(legend(r"|\omega j(\omega)|", "normalized"))
            else:
                a.set_ylabel(ylabel_list[i])
    fig.tight_layout()
    return fig, ax

def plot_j(main_df, max_freq=None):
    fig, ax = create_frame()
    
    frequencies = main_df["frequencies"]
    add_verticals(frequencies, ax, max_freq=max_freq)
    add_current_density_to_plot(main_df, ax, max_freq=max_freq)

    if max_freq is not None:
        ax.set_xlim(0, max_freq)

    fig.tight_layout()
    plt.show()
    
    
if __name__ == '__main__':
    main_df = load_panda("HHG", "test/cosine_laser/PiFlux", "current_density.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))
    plot_j(main_df)