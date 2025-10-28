import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))
def cauchy(x, mu, gamma):
    return (1. / np.pi ) * (gamma / ((x - mu)**2 + gamma**2))
def sech_distrubution(x, mu, sigma):
    return (1. / (2. * sigma)) / np.cosh(0.5 * np.pi * (x - mu) / sigma)
    
def create_frame(nrows=1, ncols=1, ylabel_list=None, **kwargs):
    fig, ax = plt.subplots(nrows=nrows, sharex=True, **kwargs)
    if (nrows == 1 and ncols==1):
        if ylabel_list is None:
            ax.set_xlabel(legend(r"t / T_\mathrm{L}"))
        else:
            ax.set_xlabel(ylabel_list)
        ax.set_ylabel(legend(r"j(t)", "normalized"))
    else:
        ax[-1].set_xlabel(legend(r"t / T_\mathrm{L}"))
        for i, a in enumerate(ax):
            if ylabel_list is None:
                a.set_ylabel(legend(r"\partial_t j(t)", "normalized"))
            else:
                a.set_ylabel(ylabel_list[i])
    #fig.tight_layout()
    return fig, ax

def add_current_density_to_plot(df, ax, label=None, sigma=None, normalize=True, substract=None, shift=0, derivative=True, t_average=50, **kwargs):
    times = np.linspace(0, df["t_end"] - df["t_begin"], len(df["current_density_time"])) / (2 * np.pi)
    __data = df["current_density_time"]
    if t_average > 0:
        __std_dev = 1e-3 * t_average * df["photon_energy"] / (2 * np.pi * 0.6582119569509065)
        __kernel = gaussian(times, times[len(times)//2], 0.5 * __std_dev)
        __data = np.convolve(__data, __kernel, mode='same')
    
    if derivative:
        __data = -np.gradient(__data, times[1]-times[0])
    
    if substract is not None:
        __data -= substract(times)
    
    normalization = np.max(np.abs(__data)) if normalize else 1.0
    __data /= normalization
    #__data *= -1 # only the magnitude matters, but the inverted sign looks better
    ax.plot(times, __data - shift, label=label, **kwargs)

    if sigma is not None:
        worst_case = 4 * sigma * normalization
        ax.fill_between(times, __data - worst_case, __data + worst_case, alpha=0.2, **kwargs)

def plot_j(df):
    fig, ax = create_frame()
    add_current_density_to_plot(df, ax)
    ax.legend()

    plt.show()

if __name__ == '__main__':
    fig, ax = create_frame()
    main_df = load_panda("HHG", "cascade_new/exp_laser/PiFlux", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=1.5e6, band_width=200, field_amplitude=1, photon_energy=1, tau_diag=10, tau_offdiag=-1, t0=0))
    add_current_density_to_plot(main_df, ax, f"Fine grid")
    main_df = load_panda("HHG", "cascade_prec/exp_laser/PiFlux", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=1.5e6, band_width=200, field_amplitude=1, photon_energy=1, tau_diag=10, tau_offdiag=-1, t0=0))
    add_current_density_to_plot(main_df, ax, f"Coarse grid", ls="--")
    
    ax.legend()
    plt.show()