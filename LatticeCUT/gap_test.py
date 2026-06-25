import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

fig, ax = plt.subplots()

N=16000
SYSTEM = 'bcc'
main_df = load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz",
                    **lattice_cut_params(N=N, 
                                         g=1.89,
                                         U=0.01, 
                                         E_F=-0.5,
                                         omega_D=0.02))

energy_space = main_df['energies']
delta = main_df['Delta']
ax.plot(energy_space, delta, 'k-')

ax.set_xlabel(r'$\epsilon - \mu$')
ax.set_ylabel(r'$\Delta$')

ax.text(0.8, 0.8, f"Max={np.max(delta):.5f}\nMin={np.min(delta):.5f}", transform=ax.transAxes, ha="right")

fig.tight_layout()

plt.show()