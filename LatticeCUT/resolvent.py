import numpy as np
import matplotlib.pyplot as plt
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

import mrock.continued_fraction as cf

SYSTEM = 'sc'
N=16000
params = lattice_cut_params(N=N, 
                            g=1.,
                            U=0., 
                            E_F=0,
                            omega_D=0.02)
main_df = load_panda("lattice_cut", f"./{SYSTEM}", "resolvents.json.gz", **params)



resolvents = cf.ContinuedFraction(main_df, ignore_first=200, ignore_last=280)
print("Delta_true = ", resolvents.continuum_edges()[0])

fig, ax = plt.subplots()
ax.set_xlabel(r"$\omega / W$")
ax.set_ylabel(r"$\mathcal{A} (\omega) / W^{-1}$")

w_lin = np.linspace(0, .5 * main_df["continuum_boundaries"][1], 10000, dtype=complex)#
w_lin += 1e-4j

A_phase = resolvents.spectral_density(w_lin, "phase_SC",     with_terminator=True)
A_higgs = resolvents.spectral_density(w_lin, "amplitude_SC", with_terminator=True)

ax.plot(w_lin.real, A_phase, label="Phase")
ax.plot(w_lin.real, A_higgs, label="Higgs")
ax.set_ylim(-0.05, 3.5)

resolvents.mark_continuum(ax)

ax.set_xlim(np.min(w_lin.real), np.max(w_lin.real))
ax.legend()
fig.tight_layout()
plt.show()