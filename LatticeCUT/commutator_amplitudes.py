import numpy as np
import matplotlib.pyplot as plt
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

import mrock_centralized_scripts.FullDiagPurger as fdp

SYSTEM = 'sc'
E_F=0
OMEGA_D = 0.02
G = 0.3
DIR = f"./{SYSTEM}"
N=16000

PICK=0

fig, ax = plt.subplots()
fig.subplots_adjust(hspace=0)
ax.set_ylabel("Summands")
ax.set_xlabel(r"$\varepsilon - \mu$")
ax.set_xlim(-0.3, 0.3)


params = lattice_cut_params(N=N, 
                            g=G,
                            U=0, 
                            E_F=E_F,
                            omega_D=OMEGA_D)
main_df = load_panda("lattice_cut", DIR, "full_diagonalization.json.gz", print_date=False, **params)
gap_df = load_panda("lattice_cut", DIR, "gap.json.gz", print_date=False, **params)
purger = fdp.FullDiagPurger(main_df, np.linspace(-1, 1, N) - main_df["chemical_potential"])

Delta = gap_df["Delta"]
epsilon = np.linspace(-1, 1, N)
xi = epsilon - gap_df["chemical_potential"]
E = np.sqrt(xi**2 + Delta**2)
nu = purger.amplitude_eigenvectors[PICK][N:]
alpha = purger.amplitude_eigenvectors[PICK][:N]
omega = purger.amplitude_eigenvalues[PICK]
rho = gap_df["dos"]
dEps = 2. / N
rho_tilde = np.sum(rho[np.abs(epsilon) <= 2. * OMEGA_D]) * dEps / (4. * OMEGA_D)

G_unit = -G / (2 * rho_tilde)

single = 2 * E**2 * nu / np.where(Delta != 0, Delta, np.inf)

def sum(k):
    mask = np.abs(k - epsilon) <= 2 * OMEGA_D
    return G_unit * dEps * np.sum(rho[mask] * E[mask] * nu[mask] / np.where(Delta[mask] != 0, Delta[mask], np.inf))
sums = 2 * np.array([ sum(k) for k in epsilon ])

ax.plot(xi, single, label="Single")
ax.plot(xi, sums, label="Sum")

#ax.plot(xi, single-sums, label="Diff")
C_k = (single + sums)

fig2, (ax2, ax3) = plt.subplots(ncols=2, sharex=True)
ax2.set_xlim(-0.3, 0.3)
ax2.set_xlabel(r"$\varepsilon - \mu$")
ax3.set_xlabel(r"$\varepsilon - \mu$")
ax2.set_ylabel(r"Conditions")
nu_condition = 2 * Delta * C_k / omega**2
ax2.plot(xi, nu_condition, label=r"$\nu$")
ax2.plot(xi, nu, ls="--")

def sum2(k):
    mask = np.abs(k - epsilon) <= 2 * OMEGA_D
    return G_unit * dEps * np.sum(rho[mask] * xi[mask] * C_k[mask] / E[mask])

alpha_condition = -2 * (xi * C_k + np.array([ sum2(k) for k in epsilon ])) / omega**2
ax3.plot(xi, alpha_condition, label=r"$\alpha$")
ax3.plot(xi, alpha, ls="--")



fig_check, ax_check = plt.subplots()
ax_check.set_xlim(-0.3, 0.3)
ax_check.set_xlabel(r"$\varepsilon - \mu$")
ax_check.set_ylabel(r"$\alpha + \xi \nu / \Delta$")
check = alpha_condition + nu_condition * xi / np.where(Delta != 0, Delta, np.inf)
ax_check.plot(xi, check)


plt.show()