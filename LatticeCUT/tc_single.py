import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *
from scipy.optimize import curve_fit

SYSTEM = 'bcc'
N=10000
params = lattice_cut_params(N=N, 
                            g=1.6,
                            U=0.0, 
                            E_F=-0.2,
                            omega_D=0.02)
main_df = load_panda("lattice_cut", f"./T_C/{SYSTEM}", "T_C.json.gz", **params)

fig, ax = plt.subplots()

def linear_model(T, m, b):
    return m*T + b

Ts = main_df['temperatures']
deltas = np.array(main_df['max_gaps'])

cut = np.min([np.argmin(np.abs(Ts - 0.95 * Ts[-1])), len(Ts) - 5])
T_fit = Ts[cut:]
y_fit = (deltas[cut:])**2

popt, pcov = curve_fit(linear_model, T_fit, y_fit)
m, b = popt
dm, db = np.sqrt(np.diag(pcov))

#print("m =", m, "+/-", dm)
#print("b =", b, "+/-", db)
#print("\nCovariance matrix:")
#print(pcov)

# Convert to A and Tc
A = np.sqrt(-m)
Tc = b / (-m)

# error propagation
dA = abs(0.5/np.sqrt(-m) * dm)
dTc = np.sqrt((b/m**2)**2 * dm**2 + (1/m**2)*db**2 + 2*(b/m**2)*(-1/m)*pcov[0,1])

print(f"\nA = {A} ± {dA}")
print(f"Tc = {Tc} ± {dTc}")

# plot check
t_lin = np.linspace(T_fit.min(), Tc, 300)
plt.plot(Ts, deltas, "x--", label="data")
plt.plot(t_lin, np.sqrt(linear_model(t_lin, *popt)), "--", label="linearized fit")
plt.axvline(T_fit[0], linestyle=":", color="k")

ax.legend()
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$\Delta_\mathrm{max}(T)$')
fig.tight_layout()
plt.show()