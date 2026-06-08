import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *
from matplotlib.lines import Line2D

SYSTEM = 'bcc'
DIR = '.'
N=10000
G=2.2
OMEGA_D=0.02
E_F=-0.5

fig, ax = plt.subplots()

main_df = load_all(f"lattice_cut/./T_C/{SYSTEM}/N={N}/", "T_C.json.gz", 
                       condition=[f"g={G}", f"E_F={E_F}", f"omega_D={OMEGA_D}"]).sort_values('U', ignore_index=True)

cmap = plt.get_cmap('inferno')
n_rows = len(main_df.index)
skip = n_rows // 6
U_max = np.max(main_df['U'])

for i, row in main_df.iterrows():
    if i % skip != 0:
        continue
    
    temperatures = row['temperatures']
    Delta_true   = row['true_gaps']
    Delta_max    = row['max_gaps']
    U = row['U']
    color = cmap(U / U_max)

    ax.plot(temperatures, Delta_max,  c=color, label=f"${U}$")
    ax.plot(temperatures, Delta_true, c=color, ls="--")

c_leg = ax.legend(loc="upper right", title="$U$")
l_leg = ax.legend([Line2D([],[], c="k", ls="-"), Line2D([],[], c="k", ls="--")],
              [r"$\Delta_\mathrm{max}$", r"$\Delta_\mathrm{true}$"],
              loc="center left")
ax.add_artist(c_leg)

ax.set_ylim(0, None)
ax.set_xlabel(r'$T / W$')
ax.set_ylabel(r'$\Delta / W$')
fig.tight_layout()
plt.show()