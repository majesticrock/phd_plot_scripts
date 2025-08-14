import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *

fig, ax = plt.subplots()

SYSTEM = 'fcc'
main_df = load_panda('lattice_cut', f'./{SYSTEM}', 'gap.json.gz',
                    **lattice_cut_params(N=16000, 
                                         g=1.5, 
                                         U=0, 
                                         E_F=0,
                                         omega_D=0.05))

energy_space = main_df['energies']
ax.plot(energy_space, main_df['Delta'], 'k-', marker='x', markevery=20)
if 'inner_min' in main_df.index:
    ax.axvline(main_df['inner_min'], ls=':', c='k', alpha=0.4)
    ax.axvline(main_df['inner_max'], ls=':', c='k', alpha=0.4)

rho_ax = ax.twinx()
rho_ax.plot(energy_space, main_df['dos'], c='red', ls='--')
rho_ax.tick_params(axis='y', colors='red')
rho_ax.yaxis.label.set_color('red')

ax.set_xlabel(r'$\epsilon - \mu$')
ax.set_ylabel(r'$\Delta$')
rho_ax.set_ylabel(r'$\rho(\epsilon)$')

fig.tight_layout()

plt.show()