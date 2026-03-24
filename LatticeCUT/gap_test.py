import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

fig, ax = plt.subplots()

SYSTEM = 'bcc'
main_df = load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz",
                    **lattice_cut_params(N=16000, 
                                         g=1.4,
                                         U=0, 
                                         E_F=0,
                                         omega_D=0.02))

energy_space = main_df['energies']
ax.plot(energy_space, main_df['Delta'], 'k-')

rho_ax = ax.twinx()
rho_ax.plot(energy_space, main_df['dos'], c='red', ls='--')
rho_ax.tick_params(axis='y', colors='red')
rho_ax.yaxis.label.set_color('red')
rho_ax.set_ylabel(r'$\rho(\epsilon)$')

print(np.sum(np.where(np.abs(energy_space) <= 0.02, main_df['dos'], 0.0)) * (energy_space[1] - energy_space[0]))

ax.set_xlabel(r'$\epsilon - \mu$')
ax.set_ylabel(r'$\Delta$')

fig.tight_layout()

plt.show()