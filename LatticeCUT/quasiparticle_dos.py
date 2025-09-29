import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *
from scipy.interpolate import interp1d

fig, ax = plt.subplots()

E_F = -0.5
SYSTEM = 'sc'
main_df = load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz",
                    **lattice_cut_params(N=16000, 
                                         g=0.5,
                                         U=0, 
                                         E_F=E_F,
                                         omega_D=0.02))

energies = main_df['energies']
sp_dos = interp1d(main_df['energies'], main_df['dos'], bounds_error=False, fill_value=0.0)
gaps = main_df['Delta']
gaps_inter = interp1d(energies, gaps, bounds_error=False, fill_value=0.0)
gaps_deriv_inter = interp1d(energies, np.gradient(gaps, energies[1] - energies[0]), bounds_error=False, fill_value=0.0)

quasiparticle_energies = np.sqrt((energies - E_F)**2 + gaps**2)
qp_inter = interp1d(energies, quasiparticle_energies, bounds_error=False, fill_value=0.0)

omegas = np.linspace(-np.max(quasiparticle_energies), np.max(quasiparticle_energies), 400)
g_pos = [ interp1d(energies, quasiparticle_energies - omega, bounds_error=False, fill_value=0.0) for omega in omegas ]
g_neg = [ interp1d(energies, -quasiparticle_energies - omega, bounds_error=False, fill_value=0.0) for omega in omegas ]

from scipy.optimize import brentq
qp_dos = np.zeros_like(g_pos)

def integrand(epsilon_0):
    return sp_dos(epsilon_0) * qp_inter(epsilon_0) / np.abs(gaps_inter(epsilon_0) * gaps_deriv_inter(epsilon_0) + epsilon_0)

for i, g in enumerate(g_pos):
    try:
        epsilon_0 = brentq(g, np.min(energies), -abs(E_F))
        qp_dos[i] += integrand(epsilon_0)
    except ValueError:
        pass
    try:
        epsilon_0 = brentq(g, abs(E_F), np.max(energies))
        qp_dos[i] += integrand(epsilon_0)
    except ValueError:
        pass
    
    if E_F != 0:
        try:
            epsilon_0 = brentq(g, -abs(E_F), 0.)
            qp_dos[i] += integrand(epsilon_0)
        except ValueError:
            pass
        try:
            epsilon_0 = brentq(g, 0., abs(E_F))
            qp_dos[i] += integrand(epsilon_0)
        except ValueError:
            pass

for i, g in enumerate(g_neg):
    try:
        epsilon_0 = brentq(g, np.min(energies), -abs(E_F))
        qp_dos[i] += integrand(epsilon_0)
    except ValueError:
        pass
    try:
        epsilon_0 = brentq(g, abs(E_F), np.max(energies))
        qp_dos[i] += integrand(epsilon_0)
    except ValueError:
        pass
    
    if E_F != 0:
        try:
            epsilon_0 = brentq(g, -abs(E_F), 0.)
            qp_dos[i] += integrand(epsilon_0)
        except ValueError:
            pass
        try:
            epsilon_0 = brentq(g, 0., abs(E_F))
            qp_dos[i] += integrand(epsilon_0)
        except ValueError:
            pass
        
ax.plot(omegas, qp_dos)

fig.tight_layout()

plt.show()