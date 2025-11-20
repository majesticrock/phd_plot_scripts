import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *

SYSTEM = 'bcc'
DIR = 'test'
N=2000
E_F=-0.5
U=0.0
params = lattice_cut_params(N=N, 
                            g=2.0,
                            U=U, 
                            E_F=E_F,
                            omega_D=0.02)
main_df = load_panda("lattice_cut", f"{DIR}/T_C/{SYSTEM}", "T_C.json.gz", **params)
gap_df = load_panda("lattice_cut", f"{DIR}/T_C/{SYSTEM}", "all_gaps.json.gz", **params)
dos_df  = load_panda("lattice_cut", f"./bcc", "gap.json.gz", **lattice_cut_params(N=N, g=0., U=0., E_F=0., omega_D=0.02))

fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(8, 4.8))

epsilon = np.linspace(-1, 1, N, endpoint=True)
temps = main_df["temperatures"]
X, Y = np.meshgrid(epsilon, temps)

def qp_dis(Delta, mu=E_F):
    return np.sqrt((epsilon - mu)**2 + Delta**2)

if "chemical_potentials" in main_df.index:
    occupation = np.array([ 0.5 * (1. - (epsilon - mu) / qp_dis(gap, mu) * ( np.tanh(qp_dis(gap, mu) / (2. * T)) if T > 0 else 1.))
                           for gap, T, mu in zip(gap_df["finite_gaps"], temps, main_df["chemical_potentials"]) ])
    pair_creation = np.array([ np.abs(gap) / (2. * qp_dis(gap, mu)) * ( np.tanh(qp_dis(gap, mu) / (2. * T)) if T > 0 else 1.) 
                           for gap, T, mu in zip(gap_df["finite_gaps"], temps, main_df["chemical_potentials"]) ])
else:
    occupation = np.array([ 0.5 * (1. - (epsilon - E_F) / qp_dis(gap) * ( np.tanh(qp_dis(gap) / (2. * T)) if T > 0 else 1.))
                           for gap, T in zip(gap_df["finite_gaps"], temps) ])
    pair_creation = np.array([ np.abs(gap) / (2. * qp_dis(gap)) * ( np.tanh(qp_dis(gap) / (2. * T)) if T > 0 else 1.) 
                           for gap, T in zip(gap_df["finite_gaps"], temps) ])

fillings = np.sum(occupation * dos_df["dos"], axis=1) * (epsilon[1] - epsilon[0])
axes[0].plot(fillings, temps)
a, b = axes[0].get_xlim()
axes[0].set_xlim(a - 0.1 * (b - a), b + 0.1 * (b - a))
axes[0].set_ylim(0, np.max(temps))
axes[0].set_title(r"$(1/N) \sum_k \langle c_{k}^{\dagger} c_{k} \rangle$")

axes[1].set_title(r"$\langle c_{k}^{\dagger} c_{k} \rangle$")
axes[2].set_title(r"$2 |\langle c_{k} c_{k} \rangle |$")

cont = axes[1].pcolormesh(X - E_F, Y,  occupation, cmap="viridis")
cont2 = axes[2].pcolormesh(X - E_F, Y, 2. * pair_creation, cmap="viridis")

cbar = fig.colorbar(cont, ax=axes[1:])
cbar.set_label("Expectation value")

axes[1].set_xlabel(r'$(\varepsilon - E_\mathrm{F}) / W$')
axes[2].set_xlabel(r'$(\varepsilon - E_\mathrm{F}) / W$')
axes[0].set_xlabel(r'Filling')
axes[0].set_ylabel(r'$T / W$')

plt.show()