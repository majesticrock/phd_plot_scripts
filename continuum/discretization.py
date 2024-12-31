import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *

MEV_FACTOR = 1e3

fig, ax = plt.subplots(figsize=(6.4, 2.4))

NK = 20000
main_df = load_panda("continuum", "offset_10", "gap.json.gz", 
                    **continuum_params(N_k=NK, T=0, coulomb_scaling=0, screening=0, k_F=4.25, g=0.5, omega_D=10))
pd_data = main_df["data"]
pd_data["ks"] /= main_df["k_F"]

ax.plot(pd_data["ks"], label="Discretization")
ax.axvline(NK // 4, ls="--", color="black")
ax.axvline(3 * NK // 4, ls="--", color="black")

ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$k / k_\mathrm{F}$")
#ax.legend()
fig.tight_layout()

plt.show()