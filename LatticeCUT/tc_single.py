import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *

SYSTEM = 'bcc'
params = lattice_cut_params(N=8000, 
                            g=1.2,
                            U=0.1, 
                            E_F=-0.5,
                            omega_D=0.02)
main_df = load_panda("lattice_cut", f"./T_C/{SYSTEM}", "T_C.json.gz", **params)

fig, ax = plt.subplots()

Ts = main_df['temperatures']
max_gaps = np.array([ gaps for gaps in main_df['max_gaps'] ])
ax.plot(Ts, max_gaps, label="Data")

from scipy.optimize import curve_fit
def fit_func(T, A, Tc):
    return A * np.sqrt(Tc - T)
cut = len(Ts) // 2
popt, pcov = curve_fit(fit_func, Ts[-cut:], max_gaps[-cut:], bounds=( [-np.inf, Ts[-1]], [np.inf, Ts[-1] + 10 * (Ts[-1] - Ts[-2])] ))
print(f"A ={popt[0]} +/- {np.sqrt(pcov[0][0])}")
print(f"Tc={popt[1]} +/- {np.sqrt(pcov[1][1])}")

t_lin = np.linspace(Ts[len(Ts) // 5], popt[1], 500)
ax.plot(t_lin, fit_func(t_lin, *popt), ls="--", label="Fit")

ax.legend()
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$\Delta_\mathrm{max}(T)$')
fig.tight_layout()
plt.show()