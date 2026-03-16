import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
E_F=0.0
N=16000

main_df = load_pickle(f"lattice_cut/sc/N={N}/", "gaps.pkl").query(
    f"E_F == {E_F} & U == 0 & omega_D == 0.02 & g > 0.2").sort_values("g")

eps = np.linspace(-1, 1, N) - E_F
delta_max = main_df["Delta_max"]
deltas = main_df["Delta"]

fwhm = np.array([
        np.abs(eps[np.argmin(np.abs(delta - np.max(delta) / 2))])
    for delta in deltas])

#ax.plot(main_df["g"], fwhm, label="FWHM")
ax.plot(main_df["g"], delta_max, label="Delta_max", c="k")
for k in [7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900]:
    ax.plot(main_df["g"], np.array([delta[k] for delta in deltas]), label=r"$\Delta_{k=" + str(k) + "}$")

ax.legend()
ax.set_xlabel("g")

plt.show()