import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
import continued_fraction_pandas as cf
import spectral_peak_analyzer as spa
from get_data import load_panda, continuum_params
from scipy.signal import find_peaks

pd_data = load_panda("continuum", "offset_20", "resolvents.json.gz",
                    **continuum_params(N_k=20000, T=0, coulomb_scaling=0, screening=1, k_F=4.25, g=0.8, omega_D=10))
resolvents = cf.ContinuedFraction(pd_data, ignore_first=30, ignore_last=90)

fig, ax = plt.subplots()
ax.set_xlabel(r"$\ln (\omega / \mathrm{meV})$")
ax.set_ylabel(r"$\ln (\Re [G] (\omega) \cdot \mathrm{eV})]$")

w_lin = np.linspace(0, 0.99999 * pd_data["continuum_boundaries"][0], 15000, dtype=complex)
w_lin += 1e-4j

__higgs = resolvents.spectral_density(w_lin, "amplitude_SC", withTerminator=True)
__phase = resolvents.spectral_density(w_lin, "phase_SC", withTerminator=True)

__higgs_indizes = find_peaks(__higgs, distance=int(1 / (w_lin[1].real - w_lin[0].real)))[0]
__higgs_positions = np.array([w_lin[i].real for i in __higgs_indizes])

peak_data = spa.analyze_peak(lambda x: resolvents.continued_fraction(x, "amplitude_SC", withTerminator=True).real,
                 lambda x: resolvents.continued_fraction(x, "amplitude_SC", withTerminator=True).imag,
                 __higgs_positions[0], lower_continuum_edge=pd_data["continuum_boundaries"][0],
                 imaginary_offset=1e-6,
                 reversed=True,
                 range=0.1,
                 begin_offset=0.1,
                 scaling=1e-3, # so we can give the range in meV rather than eV
                 plotter=ax)

ax2 = ax.twinx().twiny()
ax2.plot(1e3 * w_lin.real, __higgs, color="red")
ax2.plot(1e3 * w_lin.real, resolvents.continued_fraction(w_lin, "amplitude_SC", withTerminator=True).real, color="blue")
ax2.set_xlabel("$\\omega$ [meV]")
ax2.set_ylabel("$\\mathcal{A} (\\omega)$ [1/meV]")


print(__higgs_positions)
print(peak_data.position)


plt.show()