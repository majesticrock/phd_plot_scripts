import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
import continued_fraction_pandas as cf
import spectral_peak_analyzer as spa
from get_data import load_panda, continuum_params
from scipy.signal import find_peaks

FIT_PEAK_N = 0
MODE_TYPE = "amplitude_SC"

__INV_MEV__ = 1e-3
__MEV__ = 1e3
def is_phase_peak(peak):
    return abs(peak) < __INV_MEV__

pd_data = load_panda("continuum", "offset_10", "resolvents.json.gz",
                    **continuum_params(N_k=20000, T=0, coulomb_scaling=0, screening=1, k_F=4.25, g=0.6, omega_D=10))
resolvents = cf.ContinuedFraction(pd_data, ignore_first=80, ignore_last=90)

fig, ax = plt.subplots()
ax.set_xlabel(r"$\ln (\omega / \mathrm{meV})$")
ax.set_ylabel(r"$\ln (\Re [G] (\omega) \cdot \mathrm{eV})]$")

w_lin = np.linspace(0, pd_data["continuum_boundaries"][0], 150000, dtype=complex)
w_lin += 1e-8j

spectral = resolvents.spectral_density(w_lin, MODE_TYPE, withTerminator=True)

spectral_indizes = find_peaks(spectral)[0]
spectral_positions = np.array([w_lin[i].real for i in spectral_indizes])

spectral_real = lambda x: resolvents.continued_fraction(x, MODE_TYPE, withTerminator=True).real
spectral_imag = lambda x: resolvents.continued_fraction(x + 1e-8j, MODE_TYPE, withTerminator=True).imag

if is_phase_peak(spectral_positions[FIT_PEAK_N]):
    begin_offset = min(1, 5e2 * resolvents.continuum_edges()[0])
    range = 0.01 * min(1, 5e2 * resolvents.continuum_edges()[0])
else:
    begin_offset = 1e-7
    range = 1e-6

peak_data = spa.analyze_peak(spectral_real, spectral_imag, 
                             peak_position=spectral_positions[FIT_PEAK_N], 
                             lower_continuum_edge=pd_data["continuum_boundaries"][0],
                             reversed=False,
                             range=range,
                             begin_offset=begin_offset,
                             scaling=__INV_MEV__, # so we can give the range in meV rather than eV
                             improve_peak_position=True,#not is_phase_peak(spectral_positions[FIT_PEAK_N]),
                             plotter=ax)

ax2 = ax.twinx().twiny()
ax2.plot(__MEV__ * w_lin.real, spectral_imag(w_lin), color="red")
ax2.plot(__MEV__ * w_lin.real, spectral_real(w_lin), color="blue")
ax2.axvline(__MEV__ * peak_data.position, color="k", linestyle=":")
ax2.axvline(__MEV__ * spectral_positions[FIT_PEAK_N], color="k", linestyle="--")
ax2.set_xlabel("$\\omega$ [meV]")
ax2.set_ylabel("$\\mathcal{A} (\\omega)$ [1/meV]")
ax2.set_ylim(-5, 5)

print(peak_data)

plt.show()