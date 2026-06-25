import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *
from scipy.optimize import curve_fit

def linear_model(T, m, b):
    return m*T + b

fig, (ax, ax_ratio) = plt.subplots(nrows=2, sharex=True)

SYSTEM = 'bcc'
OMEGA_D=0.03
N=10000
U=0.
E_F=-0.5

main_df = load_all(f"lattice_cut/./T_C/{SYSTEM}/N={N}/", "T_C.json.gz", 
                   condition=[f"U={U}", f"E_F={E_F}", f"omega_D={OMEGA_D}"]).sort_values('g')
mask = main_df["temperatures"].apply(lambda arr: len(arr) >= 10)

main_df = main_df[mask].reset_index(drop=True)
interactions = main_df["g"].to_numpy(dtype=np.float64)

max_gaps = main_df["max_gaps"]
true_gaps = main_df["true_gaps"]
TCs = np.zeros_like(interactions)

for i, (_Ts, _deltas) in enumerate(zip(main_df["temperatures"], max_gaps)):
    mask = np.abs(_deltas) > 1e-8
    Ts = np.array(_Ts)[mask]
    deltas = np.array(_deltas)[mask]
    
    cut = np.min([np.argmin(np.abs(Ts - 0.95 * Ts[-1])), len(Ts) - 5])
    T_fit = Ts[cut:]
    y_fit = (deltas[cut:])**2
    
    popt, pcov = curve_fit(linear_model, T_fit, y_fit)
    m, b = popt
    TCs[i] = b / (-m)
    
ax.plot(interactions, np.gradient(TCs, interactions))

max_gaps_at_zero  = np.gradient(np.array([delta[0] for delta in max_gaps])  / TCs, interactions) 
true_gaps_at_zero = np.gradient(np.array([delta[0] for delta in true_gaps]) / TCs, interactions) 
ax_ratio.plot(interactions, max_gaps_at_zero , label=r"$\Delta_\mathrm{max}$")
ax_ratio.plot(interactions, true_gaps_at_zero, label=r"$\Delta_\mathrm{true}$")
    
ax_ratio.axhline(1.764, color="k", ls=":", label="BCS")
ax.set_ylabel(r'$T_C$')
ax.set_ylim(0, ax.get_ylim()[1])

ax_ratio.legend()
ax_ratio.set_xlabel(r'$g$')
ax_ratio.set_ylabel(r"$\Delta_0 / T_c$")

fig.tight_layout()
plt.show()