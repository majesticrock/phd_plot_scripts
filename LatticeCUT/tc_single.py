import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *

SYSTEM = 'single_peak30'
main_df = load_panda("lattice_cut", f"./T_C/{SYSTEM}", "T_C.json.gz",
                    **lattice_cut_params(N=8000, 
                                         g=1.12,
                                         U=0., 
                                         E_F=-0.5,
                                         omega_D=0.02))

fig, ax = plt.subplots()

max_gaps = np.array([ np.max(np.abs(gaps)) for gaps in main_df['finite_gaps'] ])
ax.plot(main_df['temperatures'], max_gaps)

ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$\Delta_\mathrm{max}(T)$')
fig.tight_layout()
plt.show()