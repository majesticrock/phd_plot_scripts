import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *
from scipy.optimize import curve_fit

def linear_model(T, m, b):
    return m*T + b

fig, (ax, ax_ratio) = plt.subplots(nrows=2, sharex=True)

SYSTEM = 'bcc'
OMEGA_D=0.02
N=10000
U=0.0
E_F=-0.2

main_df = load_all(f"lattice_cut/./T_C/{SYSTEM}/N={N}/", "T_C.json.gz", 
                   condition=[f"U={U}", f"E_F={E_F}", f"omega_D={OMEGA_D}"]).sort_values('g')
mask = main_df["temperatures"].apply(lambda arr: len(arr) >= 10)

main_df = main_df[mask].reset_index(drop=True)
interactions = main_df["g"].to_numpy(dtype=np.float64) - U

main_df["time"] = pd.to_datetime(main_df["time"], format="%d-%m-%Y %H:%M:%S")
cutoff = pd.to_datetime(main_df["time"].dt.year.astype(str) + "-11-21")
result = main_df[main_df["time"] < cutoff]
for _, xresult in result.iterrows():
    print(xresult["g"])

max_gaps = main_df["max_gaps"]
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
    
ax.plot(interactions, TCs)

gaps_at_zero = np.array([delta[0] for delta in max_gaps])
ax_ratio.plot(interactions, gaps_at_zero / TCs, label=f"$\\mu^*={U}$")
    
ax_ratio.axhline(1.764, color="k", ls="--", label="BCS")
ax.set_ylabel(r'$T_C$')
ax.set_ylim(0, ax.get_ylim()[1])

ax_ratio.legend()
ax_ratio.set_xlabel(r'$g-\mu^*$')
ax_ratio.set_ylabel(r"$\Delta_0 / T_c$")

fig.tight_layout()
plt.show()