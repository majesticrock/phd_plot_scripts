import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *
from scipy.optimize import curve_fit

def fit_func(T, A, Tc):
    return A * np.sqrt(Tc - T)

SYSTEM = 'single_peak30'
main_df = load_all(f"lattice_cut/./T_C/{SYSTEM}/N=8000/", "T_C.json.gz", condition="U=0.0").sort_values('g')

fig, (ax, ax_low) = plt.subplots(nrows=2, sharex=True)

max_gaps = main_df["max_gaps"]
TCs = np.zeros_like(main_df["g"].to_numpy())
TC_errors = np.zeros_like(main_df["g"].to_numpy())

for i, (Ts, deltas) in enumerate(zip(main_df["temperatures"], max_gaps)):
    cut = len(Ts) // 2
    popt, pcov = curve_fit(fit_func, Ts[-cut:], deltas[-cut:], bounds=( [-np.inf, Ts[-1]], [np.inf, Ts[-1] + 10 * (Ts[-1] - Ts[-2])] ))
    TCs[i] = popt[1]
    TC_errors[i] = np.sqrt(pcov[1][1])

ax.plot(main_df['g'], TCs, color="blue")
ax.fill_between(main_df['g'], TCs - TC_errors, TCs + TC_errors, color="blue", alpha=0.5)

ax2 = ax.twinx()
gaps_at_zero = np.array([delta[0] for delta in max_gaps])
ax2.plot(main_df['g'], gaps_at_zero, color='red')


ax.set_ylabel(r'$T_C$', color='blue')
ax.yaxis.label.set_color('blue')
ax.tick_params(axis='y', colors='blue')

ax2.set_ylabel(r"$\Delta_\mathrm{max}$", color='red')
ax2.yaxis.label.set_color('red')
ax2.tick_params(axis='y', colors='red')

ax.set_ylim(0, ax.get_ylim()[1])
ax2.set_ylim(0, ax2.get_ylim()[1])

ax_low.plot(main_df['g'], gaps_at_zero / TCs, label="Data")
ax_low.axhline(1.764, color="k", ls="--", label="BCS")

ax_low.legend()
ax_low.set_xlabel(r'$g$')
ax_low.set_ylabel(r"$\Delta_0 / T_c$")

fig.tight_layout()
plt.show()