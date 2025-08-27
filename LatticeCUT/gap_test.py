import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *

fig, ax = plt.subplots()

SYSTEM = 'bcc'
main_df = load_panda('lattice_cut', f'./{SYSTEM}', 'gap.json.gz',
                    **lattice_cut_params(N=16000, 
                                         g=1.5, 
                                         U=0, 
                                         E_F=0,
                                         omega_D=0.02))

energy_space = main_df['energies']
ax.plot(energy_space, main_df['Delta'], 'k-')
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

particle_holeness = np.sum(energy_space * main_df['Delta'] * (energy_space[1] - energy_space[0]))
print(f"Gap PHness: {particle_holeness}")

mask = (energy_space < 1 - main_df['omega_D']) & (energy_space > -1 + main_df['omega_D'])
particle_holeness = np.sum(2 * main_df['omega_D'] * main_df['dos'][mask] * main_df['dos'][mask] * energy_space[mask] * (energy_space[1] - energy_space[0]))
print(f"DOS PHness: {particle_holeness}")

fig.tight_layout()

plt.show()