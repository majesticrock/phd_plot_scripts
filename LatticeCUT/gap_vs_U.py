import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *
import color_and_linestyle_legends as cll

fig, ax = plt.subplots()

N=16000
SYSTEM = 'bcc'

main_df = load_pickle(f"lattice_cut/{SYSTEM}/N={N}", "resolvents.pkl").query(
        f"E_F==-0.5 & omega_D==0.02"
    ).sort_values("U", ignore_index=True)

df1 = main_df.query("g==1.5")
df2 = main_df.query("g==2.5")

for c, df in enumerate([df1, df2]):
    us = df["U"]
    delta_max = df["Delta_max"]
    delta_true = 0.5 * np.array([row["continuum_boundaries"][0] for _, row in df.iterrows()])
    ax.plot(us, delta_max,  c=f"C{c}")
    ax.plot(us, delta_true, c=f"C{c}", ls="--")

ax.set_xlabel(r'$U$')
ax.set_ylabel(r'$\Delta / W$')

cll.color_and_linestyle_legends(ax, color_labels=[r"$g=1.5$",r"$g=2.5$"], linestyle_labels=[r"$\Delta_\mathrm{max}$",r"$\Delta_\mathrm{true}$"])

fig.tight_layout()

plt.show()