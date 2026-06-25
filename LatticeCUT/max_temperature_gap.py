import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

fig, ax = plt.subplots()

SYSTEM = 'bcc'
DIR = '.'
N=10000
U=0.01
params = lattice_cut_params(N=N, 
                            g=1.87,
                            U=U, 
                            E_F=-0.5,
                            omega_D=0.02)
main_df = load_panda("lattice_cut", f"{DIR}/T_C/{SYSTEM}", "all_gaps.json.gz", **params)

energy_space = np.linspace(-1, 1, N)
delta_tc = main_df['finite_gaps'][-1]
delta_0 = main_df['finite_gaps'][0]

if (delta_tc[-1] > 0):
    delta_tc *= -1
if (delta_0[-1] > 0):
    delta_0 *= -1

delta_0 *= np.max(delta_tc) / np.max(delta_0)

ax.plot(energy_space, delta_tc)
ax.plot(energy_space, delta_0, ls="--")

ax.set_xlabel(r'$\epsilon - \mu$')
ax.set_ylabel(r'$\Delta$')

ax.text(0.9, 0.9, f"MaxC={np.max(delta_0):.5f}\nMinC={np.min(delta_0):.5f}", transform=ax.transAxes, ha="right")

fig.tight_layout()

plt.show()