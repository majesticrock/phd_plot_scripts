import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

fig, ax = plt.subplots()

SYSTEM = 'bcc'
main_df = load_panda("lattice_cut", f"test/{SYSTEM}", "gap_complex.json.gz",
                    **lattice_cut_params(N=4000, 
                                         g=3,
                                         U=0.1, 
                                         E_F=-0.5,
                                         omega_D=0.02))

energy_space = main_df['energies']

delta_r = main_df['Delta_real']
delta_i = main_df['Delta_imag']

delta_abs = np.sqrt(delta_r**2 + delta_i**2)
delta_phase = np.arctan2(delta_r, delta_i)

ax.plot(energy_space, delta_abs, label="Abs")
ax.plot(energy_space, delta_phase / np.pi, label="Phase")

ax.legend()

ax.set_xlabel(r'$\epsilon - \mu$')
ax.set_ylabel(r'$\Delta$')

fig.tight_layout()

plt.show()