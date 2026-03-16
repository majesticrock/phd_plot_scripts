import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

fig, axes = plt.subplots(nrows=2, sharex=True)

SYSTEM = 'sc'
N=16000
main_df = load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz",
                    **lattice_cut_params(N=N, 
                                         g=1.5,
                                         U=0, 
                                         E_F=0,
                                         omega_D=0.02))

epsilon = np.linspace(-1, 1, N)
delta = main_df['Delta']
E = np.sqrt(epsilon**2 + np.abs(delta)**2)

def pair_creation(eps, d):
    return 0.5 * d / np.sqrt(eps**2 + np.abs(d)**2)

base = pair_creation(epsilon, delta)

amp = pair_creation(epsilon, delta * 1.2) - base
axes[0].plot(epsilon, amp / np.max(amp))
comp = delta**2 / E**2
comp /= np.max(comp)
axes[0].plot(epsilon, comp, ls="--")

img = (pair_creation(epsilon, delta * np.exp(1e-1j)).imag)
axes[1].plot(epsilon, img / np.max(img))
axes[1].plot(epsilon, comp, ls="--")

axes[1].set_xlabel(r'$\epsilon - \mu$')

fig.tight_layout()

fig2, ax2 = plt.subplots()
#ax2.plot(epsilon, np.sqrt(epsilon**2 + np.abs(delta)**2 ))
ax2.plot(epsilon, 1 / np.sqrt(epsilon**2 + np.abs(delta)**2)**2, ls="--")

plt.show()