import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

fig, ax = plt.subplots()

N=16000
SYSTEM = 'bcc'
main_df = load_pickle(f"lattice_cut/{SYSTEM}/N={N}", "resolvents.pkl").query(
        f"E_F==-0.5 & omega_D==0.02 & U==0"
    ).sort_values('g', ignore_index=True)


cmap = plt.get_cmap("magma")
true_gaps = np.zeros(len(main_df.index))
for i, row in main_df.iterrows():
    true_gaps[i] = 0.5 * row["continuum_boundaries"][0]

ax.plot(main_df["g"], true_gaps)


ax.set_xlabel(r'$(\epsilon - E_\mathrm{F}) / W$')
ax.set_ylabel(r'$\Delta / W$')

fig.tight_layout()

plt.show()