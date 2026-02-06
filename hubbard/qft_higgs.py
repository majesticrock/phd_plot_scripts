import matplotlib.pyplot as plt
import numpy as np
import gzip
from gftool import sc_dos

import mrock_centralized_scripts.path_appender as __ap
__ap.append()
import continued_fraction as cf

GAP = 0.410618
T = 0.0
U = -2.5
V = -0.1

DIR_NAME = f"../raw_data_phd/pre_pandas/modes/cube/dos_6000/T={T}/U={U}/V={V}"

with gzip.open(f"{DIR_NAME}/one_particle.dat.gz", 'rt') as f_open:
    one_particle = np.loadtxt(f_open).flatten()
    one_particle = one_particle[one_particle > 0]

fig, ax = plt.subplots()
cache_sc_dos = sc_dos(np.linspace(0., 1., len(one_particle)))

def summand(eps, omega, dos):
    return 2 * GAP * GAP * dos[eps] / 6 / (one_particle[eps] * (4 * one_particle[eps]**2 - omega**2))

omega_space = np.linspace(0, 13, 1000) + 5e-3j
eps_indizes = np.arange(0, len(one_particle))
dE = 1 / len(one_particle)

sums = []
for omega in omega_space:
    sums.append(dE * np.sum(summand(eps_indizes, omega, cache_sc_dos)))
np_sums = np.array(sums)
ax.plot(omega_space.real, -6* (1. / ((-1./U) + np_sums)).imag, label=r"QFT")

data, data_real, w_lin, resolvent = cf.resolvent_data(f"{DIR_NAME}", "higgs_SC", 0., np.max(omega_space.real), 
                                                            number_of_values=2000, xp_basis=True, imaginary_offset=1e-5, messages=False)
ax.plot(w_lin, data, label=r"iEoM", ls="--")

ax.set_xlabel(r"$\omega/t$")
ax.set_ylabel(r"$\mathcal{A}_\mathrm{SC}(\omega)$")
ax.axvline(2 * GAP, color='k', linestyle='--', label=r"$2 \Delta$")

fig.tight_layout()
plt.show()