import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *
from scipy.optimize import curve_fit

def find_tc_fit(T, A, Tc):
    return A * np.sqrt(Tc - T)

fig, (ax, ax_ratio) = plt.subplots(nrows=2, sharex=True)

SYSTEM = 'bcc'
N=10000
for U, lambda_c, marker in zip([0., 0.05, 0.1], [1.86, 1.8, 1.8], ["x", "v", "o"]):
    main_df = load_all(f"lattice_cut/./T_C/{SYSTEM}/N={N}/", "T_C.json.gz", condition=[f"U={U}", "E_F=-0.5"]).sort_values('g')
    mask = main_df["temperatures"].apply(lambda arr: len(arr) >= 5)
    main_df = main_df[mask].reset_index(drop=True)

    interactions = main_df["g"].to_numpy(dtype=np.float64)# - U
    max_gaps = main_df["max_gaps"]
    TCs = np.zeros_like(interactions)
    TC_errors = np.zeros_like(interactions)

    for i, (Ts, deltas) in enumerate(zip(main_df["temperatures"], max_gaps)):
        cut = len(Ts) // 2
        popt, pcov = curve_fit(find_tc_fit, Ts[-cut:], deltas[-cut:], bounds=( [-np.inf, Ts[-1]], [np.inf, Ts[-1] + 10 * (Ts[-1] - Ts[-2])] ))
        TCs[i] = popt[1]
        TC_errors[i] = np.sqrt(pcov[1][1])

    ax.plot(interactions, TCs, marker=marker)
    mask = (interactions >= lambda_c) & (interactions < 2.1 - U)

    Tc_before_critical = TCs[mask][0]
    def critical_fit(x, lambda_c, a, b, c):
        #return a * np.where(x > lambda_c, np.sqrt(x - lambda_c), 0.0) 
        return np.where(x > lambda_c, a  / np.abs(np.log(x - lambda_c) + b), 0.0) + Tc_before_critical + c

    #popt, pcov = curve_fit(critical_fit, interactions[mask], TCs[mask], 
    #                       bounds=([lambda_c - 0.05, -np.inf, -np.inf, -np.inf], [lambda_c + 0.05, np.inf, np.inf, np.inf]))
    #for i in range(len(popt)):
    #    print(f"{popt[i]} +/- {np.sqrt(pcov[i][i])}")
    #l_lin = np.linspace(popt[0], 2.2, 2500)
    #ax.plot(l_lin, critical_fit(l_lin, *popt), ls="--")
    
    gaps_at_zero = np.array([delta[0] for delta in max_gaps])
    ax_ratio.plot(interactions, gaps_at_zero / TCs, label=f"$\\mu^*={U}$", marker=marker)
    
ax_ratio.axhline(1.764, color="k", ls="--", label="BCS")
ax.set_ylabel(r'$T_C$')
ax.set_ylim(0, ax.get_ylim()[1])

ax_ratio.legend()
ax_ratio.set_xlabel(r'$g-\mu^*$')
ax_ratio.set_ylabel(r"$\Delta_0 / T_c$")

fig.tight_layout()
plt.show()