import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *
from scipy.optimize import curve_fit

def fit_func(T, A, Tc):
    return A * np.sqrt(Tc - T)

SYSTEM = 'single_peak30'
main_df = load_all(f"lattice_cut/./T_C/{SYSTEM}/N=8000/", "T_C.json.gz", condition="U=0.0")

fig, ax = plt.subplots()

max_gaps = np.array( [np.max(np.abs(gap), axis=1) for gap in main_df['finite_gaps']], dtype=object )
TCs = np.zeros_like(main_df["g"].to_numpy())

for Ts, Delta in zip(main_df["temperatures"], max_gaps):
    print(len(Ts), len(max_gaps))

    cut = len(Ts) // 2
    popt, pcov = curve_fit(fit_func, Ts[-cut:], max_gaps[-cut:], bounds=( [-np.inf, Ts[-1]], [np.inf, Ts[-1] + 10 * (Ts[-1] - Ts[-2])] ))
    TCs = popt[1]
    TC_erros = np.sqrt(pcov[1][1])

TCs = np.array([ Ts[-1] for Ts in main_df['temperatures'] ])
ax.plot(main_df['g'], TCs, color="blue")

ax2 = ax.twinx()

ax2.plot(main_df['g'], max_gaps, color='red')

ax.set_xlabel(r'$g$')

ax.set_ylabel(r'$T_C$', color='blue')
ax.yaxis.label.set_color('blue')
ax.tick_params(axis='y', colors='blue')

ax2.set_ylabel(r"$\Delta_\mathrm{max}$", color='red')
ax2.yaxis.label.set_color('red')
ax2.tick_params(axis='y', colors='red')

fig.tight_layout()
plt.show()