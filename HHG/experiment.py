import numpy as np
import matplotlib.pyplot as plt

import current_density_fourier as cdf

import path_appender
path_appender.append()
from get_data import *
from legend import *

MODEL = "PiFlux"
v_F = 1.5e5
W = 400

df_A = load_panda("HHG", f"exp_base/expA_laser/{MODEL}", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=v_F, band_width=W, field_amplitude=1., photon_energy=1., tau_diag=30, tau_offdiag=-1, t0=0))
df_B = load_panda("HHG", f"exp_base/expB_laser/{MODEL}", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=v_F, band_width=W, field_amplitude=1., photon_energy=1., tau_diag=30, tau_offdiag=-1, t0=0))

signal_A = cdf.compute_current_density(df_A)
signal_B = cdf.compute_current_density(df_B)

def combined_signal(omega, t0):
    if t0 > 0:
        return signal_A + signal_B * np.exp(1.0j * omega * t0)
    else:
        return signal_A * np.exp(-1.0j * omega * t0) + signal_B

fig, axes = cdf.create_frame(nrows=3, figsize=(6.4, 8), 
                             ylabel_list=[legend(r"|\omega j_\mathrm{lin}(\omega) |", "a.b."), 
                                          legend(r"|\omega j_\mathrm{sim}(\omega) |", "a.b."), 
                                          legend(r"|\omega (j_\mathrm{sim}(\omega) - j_\mathrm{lin}(t))|", "a.b.")])

frequencies = df_A["frequencies"]
cdf.add_verticals(frequencies, axes[0], max_freq=40, positions='odd' if MODEL != "Honeycomb" else 'even')
cdf.add_verticals(frequencies, axes[1], max_freq=40, positions='odd' if MODEL != "Honeycomb" else 'even')
cdf.add_verticals(frequencies, axes[2], max_freq=40, positions='odd' if MODEL != "Honeycomb" else 'even')

axes[0].set_xlim(0, 40)

for i, t0 in enumerate([0, 0.5, 1, 2, 3, 4, 6]):
    main_df = load_panda("HHG", f"exp_{t0}/exp_laser/{MODEL}", "current_density.json.gz", 
                        **hhg_params(T=300, E_F=118, v_F=v_F, band_width=W, field_amplitude=1., photon_energy=1., tau_diag=30, tau_offdiag=-1, t0=0))
    
    t0_unitless = t0 * main_df["photon_energy"] / (0.6582119569509065) # the number is hbar in meV * ps
    frequencies2 = main_df["frequencies"]
    plt_signal = 2 * np.abs(combined_signal(frequencies, t0_unitless))
    axes[0].plot(frequencies, 10**(-i) * plt_signal, label=f"$t_0 = {t0}$ ps")
    cdf.add_current_density_to_plot(main_df, axes[1], max_freq=40, label=f"Proper $t_0 = {t0}$ ps", normalize=False, shift=10**(-i))
    cdf.add_current_density_to_plot(main_df, axes[2], max_freq=40, substract=combined_signal(frequencies, t0_unitless), label=f"NL $t_0 = {t0}$ ps", normalize=False, shift=10**(-i))

axes[0].legend(loc='upper right')


fig.tight_layout()
plt.show()