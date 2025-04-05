import numpy as np
import matplotlib.pyplot as plt

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

main_df = load_panda("FHHG", "test/continuous_laser", "debug.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e3, band_width=5, field_amplitude=1.6, photon_energy=5.25))

fig, ax = plt.subplots()

ys = ["j_k_0", "j_k_1", "j_k_2", "j_k_3"]

for i, y in enumerate(ys):
    ax.plot(main_df["k_zs"], main_df[y], ls="-", c=f"C{i}")
    diff = main_df[y] - main_df[y][::-1]
    ax.plot(main_df["k_zs"], diff, ls="--", c=f"C{i}")
    print(np.trapz(diff[:len(diff)//2], main_df["k_zs"][:len(diff)//2]))
    
ax.set_xlabel(legend(r"v_F k_z / \omega_L"))
ax.set_ylabel(legend(r"j"))
fig.tight_layout()

plt.show()