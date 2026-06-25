import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

N=16000
SYSTEM = 'bcc'

main_df = load_pickle(f"lattice_cut/{SYSTEM}/N={N}", "gaps.pkl").query(
        f"E_F==-0.5 & omega_D==0.02 & U==0.1 & g > 1.5"
    ).sort_values("g", ignore_index=True)

ef_idx = N//4
enh_idx = N//2

delta_ef =  np.array([ abs(row["Delta"][ef_idx])  for _, row in main_df.iterrows() ])
delta_enh = np.array([ abs(row["Delta"][enh_idx]) for _, row in main_df.iterrows() ])
g = main_df["g"]

        
fig, axes = plt.subplots(nrows=2, sharex=True)

grad_ef  = np.gradient(delta_ef, g)
grad_enh = np.gradient(delta_enh, g)

axes[0].plot(g, grad_ef)
axes[0].plot(g, grad_enh)

axes[1].plot(g, delta_ef)
axes[1].plot(g, delta_enh)

pos = g[np.argmax(grad_ef)]

for ax in axes:
    ax.axvline(pos, c="k", ls="--")
axes[0].text(pos, 0.9 * axes[0].get_ylim()[1], pos, va="top", ha="right")
print(pos)

plt.show()