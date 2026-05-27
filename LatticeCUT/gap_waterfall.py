import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

fig, ax = plt.subplots()

N=16000
SYSTEM = 'bcc'
main_df = load_pickle(f"lattice_cut/{SYSTEM}/N={N}", "gaps.pkl").query(
        f"E_F==-0.5 & omega_D==0.02 & U==0.1 & g>1.8 & g<2"
    ).sort_values('g', ignore_index=True)

filtered_df = main_df#[np.isclose(main_df["g"] * 10, np.round(main_df["g"] * 10))].reset_index(drop=True)

eps = np.linspace(-1, 1, N)

cmap = plt.get_cmap("magma")
n_lines = len(filtered_df.index)
xi = eps + 0.5

for i, row in filtered_df.iterrows():
    delta = row["Delta"]
    if delta[-1] > 0:
        delta *= -1
    if delta[N//2] > 0:
        print("Enhancement dominates at g =", row["g"])
    ax.plot(xi, row['Delta'] , c=cmap(i / n_lines))#+ 0.3 * i / n_lines


ax.set_xlabel(r'$(\epsilon - E_\mathrm{F}) / W$')
ax.set_ylabel(r'$\Delta / W$')

fig.tight_layout()

plt.show()