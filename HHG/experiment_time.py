import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import current_density_time as cdt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

from scipy.fft import rfft, rfftfreq

DIR = "exp_cl1"
MODEL = "PiFlux"
v_F = 1.5e6
W = 400

df_A = load_panda("HHG", f"{DIR}_base/expA_laser/{MODEL}", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=v_F, band_width=W, field_amplitude=1., photon_energy=1., decay_time=100))
df_B = load_panda("HHG", f"{DIR}_base/expB_laser/{MODEL}", "current_density.json.gz", 
                    **hhg_params(T=300, E_F=118, v_F=v_F, band_width=W, field_amplitude=1., photon_energy=1., decay_time=100))

times = np.linspace(0, df_A["t_end"] - df_A["t_begin"], len(df_A["current_density_time"])) / (2 * np.pi)

signal_A = df_A["current_density_time"]
signal_B = df_B["current_density_time"]
signal_AB = signal_A + signal_B

inter_A = interp1d(times, signal_A, fill_value=0.0, bounds_error=False)
inter_B = interp1d(times, signal_B, fill_value=0.0, bounds_error=False)

def combined_inter(t, t0):
    if t0 >= 0:
        A = inter_A(t)
        B = inter_B(t - t0)
    else:
        A = inter_A(t + t0)
        B = inter_B(t)
    return A + B

fig, axes = cdt.create_frame(nrows=3, figsize=(6.4, 8), 
                             ylabel_list=[legend(r"j_\mathrm{lin}(t)", "a.b."), 
                                          legend(r"j_\mathrm{sim}(t)", "a.b."), 
                                          legend(r"j_\mathrm{sim}(t) - j_\mathrm{lin}(t)", "a.b.")])
#fig_laser, ax_laser = plt.subplots()
fig_fft, axes_fft = plt.subplots(nrows=3, figsize=(6.4, 8), sharex=True)
for ax in axes_fft:
    ax.set_yscale('log')
    ax.set_xlim(0, 40)

inter_laser_A = interp1d(times, df_A["laser_function"], fill_value=0.0, bounds_error=False)
inter_laser_B = interp1d(times, df_B["laser_function"], fill_value=0.0, bounds_error=False)
def combined_laser(t, t0):
    if t0 >= 0:
        A = inter_laser_A(t)
        B = inter_laser_B(t - t0)
    else:
        A = inter_laser_A(t + t0)
        B = inter_laser_B(t)
    return A + B

for i, t0 in enumerate([0, 0.5, 1, 2, 3, 4, 6]):
    main_df = load_panda("HHG", f"{DIR}_{t0}/exp_laser/{MODEL}", "current_density.json.gz", 
                        **hhg_params(T=300, E_F=118, v_F=v_F, band_width=W, field_amplitude=1., photon_energy=1., decay_time=100))
    times2 = np.linspace(0, main_df["t_end"] - main_df["t_begin"], len(main_df["current_density_time"])) / (2 * np.pi)
    
    t0_unitless = t0 * main_df["photon_energy"] / (2*np.pi * 0.6582119569509065) # the number is hbar in meV * ps
    plot_data_combined = combined_inter(times2, t0_unitless)
    axes[0].plot(times2, plot_data_combined, label=f"$t_0 = {t0}$ ps")
    cdt.add_current_density_to_plot(main_df, axes[1], normalize=False)
    cdt.add_current_density_to_plot(main_df, axes[2], substract=lambda t: combined_inter(t, t0_unitless), label=f"$t_0 = {t0}$ ps", normalize=False)
    
    n = len(times2) * 4
    dt = times2[1] - times2[0]
    freqs_scipy = rfftfreq(n, dt)
    
    fftplot = np.abs(freqs_scipy * rfft(plot_data_combined, n))
    axes_fft[0].plot(freqs_scipy, 10**(-i) * fftplot , label=f"$t_0 = {t0}$ ps")
    fftplot = np.abs(freqs_scipy * rfft(main_df["current_density_time"], n))
    axes_fft[1].plot(freqs_scipy, 10**(-i) * fftplot )
    fftplot = np.abs(freqs_scipy * rfft(main_df["current_density_time"] - plot_data_combined, n))
    axes_fft[2].plot(freqs_scipy, 10**(-i) * fftplot )

    #ax_laser.plot(times2, combined_laser(times2, t0_unitless), c=f"C{i}")
    #ax_laser.plot(times2, main_df["laser_function"], ls="--", c=f"C{i}", linewidth=4)

axes[0].legend(loc='upper right')
#axes[1].legend(loc='upper right')
#axes[2].legend(loc='upper right')

axes_fft[0].legend(loc='upper right')
fig_fft.tight_layout()

fig.tight_layout()


plt.show()