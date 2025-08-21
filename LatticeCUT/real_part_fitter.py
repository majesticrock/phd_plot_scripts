import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
import continued_fraction_pandas as cf
import spectral_peak_analyzer as spa
from get_data import *
from scipy.signal import find_peaks

FIT_PEAK_N = 0
MODE_TYPE = "phase_SC"

def is_phase_peak(peak):
    return abs(peak) < 1e-3

SYSTEM = "sc"#"sc"#"free_electrons3"
main_df = load_panda("lattice_cut", f"./{SYSTEM}", "resolvents.json.gz",
                    **lattice_cut_params(N=16000, 
                                         g=2.45,
                                         U=0, 
                                         E_F=0,
                                         omega_D=0.02))
resolvents = cf.ContinuedFraction(main_df, ignore_first=70, ignore_last=150)

fig, ax = plt.subplots()
ax.set_xlabel(r"$\ln (\omega / t)$")
ax.set_ylabel(r"$\ln (\Re [G] (\omega) / t^{-1})]$")

w_lin = np.linspace(0, main_df["continuum_boundaries"][0], 15000, dtype=complex)
w_lin += 1e-5j


spectral = resolvents.spectral_density(w_lin, MODE_TYPE, withTerminator=True)

find_peaks_result = find_peaks(spectral, prominence=0.1)
spectral_indizes = find_peaks_result[0]
spectral_positions = np.array([w_lin[i].real for i in spectral_indizes])
print(spectral_positions)

spectral_real = lambda x: resolvents.continued_fraction(x, MODE_TYPE, withTerminator=True).real
spectral_imag = lambda x: resolvents.continued_fraction(x + 1e-8j, MODE_TYPE, withTerminator=True).imag

if is_phase_peak(spectral_positions[FIT_PEAK_N]):
    begin_offset = spectral_positions[FIT_PEAK_N] + 5e-3
    range = 0.01
else:
    begin_offset = 1e-4
    range = 1e-5

peak_data = spa.analyze_peak(spectral_real, spectral_imag, 
                             peak_position=spectral_positions[FIT_PEAK_N], 
                             lower_continuum_edge=main_df["continuum_boundaries"][0],
                             reversed=False,
                             range=range,
                             begin_offset=begin_offset,
                             improve_peak_position=True,
                             plotter=ax,
                             peak_pos_range=0.001)

ax2 = ax.twinx().twiny()
ax2.plot(w_lin.real, spectral_imag(w_lin), color="red")
ax2.plot(w_lin.real, spectral_real(w_lin), color="blue")
ax2.axvline(peak_data.position, color="k", linestyle=":")
ax2.axvline(spectral_positions[FIT_PEAK_N], color="k", linestyle="--")
ax2.set_xlabel("$\\omega / t$")
ax2.set_ylabel("$\\mathcal{A} (\\omega) / t^{-1}$")
ax2.set_ylim(-5, 5)

print(peak_data)

plt.show()