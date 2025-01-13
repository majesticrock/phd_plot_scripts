import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

OMEGA_D = 0.1 # eV
K_F = 4.25
LAMBDA = 1e-4
SCREENING = LAMBDA * 0.4107320221286488672 / np.sqrt(K_F)
K_MAX = 1e16
EPS = 1e-8

def coulomb_fock(x):
    if LAMBDA == 0:
        return 0
    k_diff = x - 1
    k_sum = x + 1
    ln_factor = (SCREENING * SCREENING + 1 - x * x) / (2.0 * x)
    log_expression = np.log((SCREENING * SCREENING + k_sum * k_sum) / (SCREENING * SCREENING + k_diff * k_diff))
    return -0.12652559550141668 * K_F * (
        1 + ln_factor * log_expression + SCREENING * (np.arctan(k_diff / SCREENING) - np.arctan(k_sum / SCREENING))
    )

E_F = 0.5 * K_F**2 + coulomb_fock(1) # eV
RHO_F = K_F / (2 * np.pi**2)
M_SQUARED = 0.5 * OMEGA_D / RHO_F # yields Delta_max = 2.3 meV
G_TILDE = K_F * M_SQUARED / (np.pi**2)
omega_tilde = 2 * OMEGA_D / (K_F**2)

def bare_dispersion(x):
    return 0.5 * K_F**2 * ( x**2 - 1 ) 

def phonon_fock(x):
    sqrt_plus = np.sqrt(x*x + omega_tilde)
    sqrt_minus = np.sqrt(np.abs(x*x - omega_tilde))

    term1 = -0.5 * sqrt_plus  * np.log( (np.abs(1 - sqrt_plus) + EPS)  / ( np.abs(1 + sqrt_plus )) )
    term2 = np.where(x*x > omega_tilde, 
                        0.5 * sqrt_minus * np.log((np.abs(1 - sqrt_minus) + EPS) / (np.abs(1 + sqrt_minus) )), 
                        sqrt_minus * np.arctan(2 / (sqrt_minus))) 
    return G_TILDE * (term1 + term2)

def renormalization_cut(x):
    sqrt_minus = np.sqrt(np.abs(x*x - omega_tilde))
    return G_TILDE * (np.where(x*x > omega_tilde, 
                              0., #-0.5 * sqrt_minus * np.log(np.abs( (K_MAX - sqrt_minus) / (K_MAX + sqrt_minus) + EPS )), 
                              sqrt_minus * np.arctan(2 / sqrt_minus)) ) #- G_TILDE * K_MAX

print("Phononic Fock energy at singularity =", phonon_fock(np.sqrt(1 - omega_tilde)), "eV")
print("Integral on paper =", integrate.quad(phonon_fock, 0.5, 1.))
fig, ax = plt.subplots(ncols=1, sharey=True)

OFFSET=0.02
x = np.linspace(1 - OFFSET, 1 + OFFSET, 10000)
#ax.plot(x, renormalization_cut(x), label="Renorm. CUT")
#ax.plot(x, bare_dispersion(x), label=r"$\epsilon_0$")
ax.plot(x, bare_dispersion(x) + phonon_fock(x) - renormalization_cut(x), label=r"$\epsilon_\mathrm{Fock}^\mathrm{Ph}$")
#ax.plot(x, bare_dispersion(x) + phonon_fock(x) - renormalization_cut(x), label=r"$\epsilon$")
#ax.plot(x, coulomb_fock(x) - coulomb_fock(1), label=r"$\epsilon_\mathrm{Fock}^\mathrm{C} (\lambda=10^{-4})$")
#twinx = ax.twinx()
#twinx.plot(x, np.where(bare_dispersion(x) != 0, np.abs((coulomb_fock(x) - coulomb_fock(1)) / bare_dispersion(x)), 0), "r--")
ax.set_xlabel(r"$x = k / k_\mathrm{F}$")
ax.set_ylabel(r"$\epsilon (x) [eV]$")
ax.grid()
#ax.legend() 
fig.tight_layout()
fig.subplots_adjust(wspace=0)
plt.show()