import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
import continued_fraction_pandas as cf
from get_data import load_panda, continuum_params
from legend import *
from ez_fit import ez_linear_fit
from uncertainties import ufloat

pd_data = load_panda("continuum", "offset_10", "resolvents.json.gz", 
                    **continuum_params(N_k=20000, T=0, coulomb_scaling=1, screening=1e-4, k_F=4.25, g=1, omega_D=10))
resolvents = cf.ContinuedFraction(pd_data, ignore_first=70, ignore_last=90)

print("Gap of the system: ", pd_data["continuum_boundaries"][0] / 2)

fig, ax = plt.subplots()
ax.set_xlabel(log_legend(r"\omega", "meV"))
ax.set_ylabel(log_legend(r"\mathcal{A}", "eV", -1))

lower_continuum_edge = pd_data["continuum_boundaries"][0] # 2 Delta
offset = 1e-3
upper = offset + 0.001

w_log = np.linspace(np.log(offset), np.log(upper), 300, dtype=complex)
spectral_higgs = resolvents.spectral_density(lower_continuum_edge + np.exp(w_log), "amplitude_SC")
spectral_phase = resolvents.spectral_density(lower_continuum_edge + np.exp(w_log), "phase_SC")

spectral = np.log(spectral_higgs )

ax.plot(w_log, spectral, label="Data")
popt, pcov, line = ez_linear_fit(w_log, spectral, ax, label="Fit", ls="--")

u_slope  = ufloat(popt[0], np.sqrt(pcov[0][0]))
u_offset = ufloat(popt[1], np.sqrt(pcov[1][1]))

print(u_slope, u_offset)

ax.legend()
fig.tight_layout()
plt.show()


##### Results:
# No Coulomb:
# Slope = -1 => A(omega) = 1/