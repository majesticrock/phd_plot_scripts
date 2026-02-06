import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

fig, ax = plt.subplots()

PARAMS = {
    'N': 4000, 
    'g': 1.5,
    'U': 0,
    'E_F': 0,
    'omega_D': 0.05
}

SYSTEM = 'sc'
main_df = load_panda("lattice_cut", f"mkl/{SYSTEM}", "gap.json.gz", **lattice_cut_params(**PARAMS))
ax.plot(main_df['energies'], main_df['Delta'], label="MKL")

main_df = load_panda("lattice_cut", f"compare_mkl/{SYSTEM}", "gap.json.gz", **lattice_cut_params(**PARAMS))
ax.plot(main_df['energies'], main_df['Delta'], label="No MKL", ls="--")

ax.set_xlabel(r'$\epsilon - \mu$')
ax.set_ylabel(r'$\Delta$')
fig.tight_layout()

# --- Resolvent comparison ---
fig_res, ax_res = plt.subplots()

import continued_fraction_pandas as cf

main_df_res = load_panda("lattice_cut", f"mkl/{SYSTEM}", "resolvents.json.gz", **lattice_cut_params(**PARAMS))
resolvent_mkl = cf.ContinuedFraction(main_df_res)
w_lin = np.linspace(-0.005 * main_df_res["continuum_boundaries"][1], 1.1 * main_df_res["continuum_boundaries"][1], 5000, dtype=complex)
w_lin += 1e-5j
A_mkl = resolvent_mkl.spectral_density(w_lin, "phase_SC", withTerminator=True)
ax_res.plot(w_lin.real, A_mkl, label="MKL")

main_df_res_compare = load_panda("lattice_cut", f"compare_mkl/{SYSTEM}", "resolvents.json.gz", **lattice_cut_params(**PARAMS))
resolvent_nomkl = cf.ContinuedFraction(main_df_res_compare)
A_nomkl = resolvent_nomkl.spectral_density(w_lin, "phase_SC", withTerminator=True)
ax_res.plot(w_lin.real, A_nomkl, label="No MKL", ls="--")

ax_res.set_xlabel(r"$\omega [\mathrm{meV}]$")
ax_res.set_ylabel(r"$\mathcal{A} (\omega) [\mathrm{eV}^{-1}]$")
ax_res.legend()
fig_res.tight_layout()

plt.show()