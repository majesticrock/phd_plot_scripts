import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *
import numpy as np

N       = 16000
OMEGA_D = 0.02
G       = 3.0
U       = 0
E_F     = -0.5
SYSTEM  = 'sc'

main_df = load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz",
                        **lattice_cut_params(N=N, 
                                             g=G,
                                             U=U, 
                                             E_F=E_F,
                                             omega_D=OMEGA_D))
Delta = main_df['Delta']
mu = main_df['chemical_potential']
xi = np.linspace(-1, 1, N) - mu
E = np.sqrt(xi**2 + Delta**2)

dDelta_dxi = np.gradient(Delta, xi)
# dE/dxi
dE_dxi = (xi + Delta * dDelta_dxi) / E

# coherence factors
u2 = 0.5 * (1.0 + xi / E)
v2 = 0.5 * (1.0 - xi / E)

sign_change = np.where(np.diff(np.sign(dE_dxi)) != 0)[0]

print("Possible stationary points:")
for i in sign_change:
    xi_star = xi[i]
    E_star = E[i]
    print(f"xi ~ {xi_star:.6g},  E ~ {E_star:.6g}")


fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

# Delta and E
ax[0].plot(xi, Delta, label='Delta(xi)')
ax[0].plot(xi, E, label='E(xi)')
ax[0].set_ylabel('Energy')
ax[0].legend()
ax[0].set_title('Gap and quasiparticle dispersion')

# derivative
ax[1].plot(xi, dE_dxi, label='dE/dxi')
ax[1].axhline(0, color='k', ls='--')

for i in sign_change:
    ax[1].axvline(xi[i], color='r', alpha=0.4)

ax[1].set_ylabel('dE/dxi')
ax[1].legend()
ax[1].set_title('Stationary-point diagnostic')

ax[2].plot(xi, u2, label='$u^2(\\xi)$')
ax[2].plot(xi, v2, label='$v^2(\\xi)$')
ax[2].set_ylabel('Coherence factors')
ax[2].legend()
ax[2].set_title('Coherence factors')

plt.tight_layout()
plt.show()