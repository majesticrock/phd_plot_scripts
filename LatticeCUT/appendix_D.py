import numpy as np
import matplotlib.pyplot as plt
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *
from mrock_centralized_scripts.create_figure import *
import mrock_centralized_scripts.FullDiagPurger as fdp

N = 16000
E_F = 0
OMEGA_D = 0.02
OMEGA_D_UNIT = 2 * OMEGA_D
G = 1
SYSTEM = 'sc'

epsilon_full = np.linspace(-1, 1, N) - E_F

BCS = True

colors = [
    "#000000",
    "#F97316",
    "#5105D3",
    "#5DDFFF",
]

fig, ax = create_normal_figure(sharey="row", sharex="row", height_to_width_ratio=3.6/6.4)
ax.set_xlabel(r"$\varepsilon / W$")

def G_unit(g, dos, epsilon):
    rho_tilde = np.sum(dos[np.abs(epsilon) <= OMEGA_D_UNIT]) / (2 * OMEGA_D_UNIT)
    return -g / (2 * rho_tilde)

def get_mask(epsilon, epsilon_k):
    if BCS:
        if abs(epsilon_k) > OMEGA_D_UNIT:
            return np.zeros_like(epsilon) > 100
        return np.abs(epsilon) <= OMEGA_D_UNIT
    else:
        return np.abs(epsilon - epsilon_k) <= OMEGA_D_UNIT 

def single_q_sum(epsilon_k, alpha, nu, epsilon, Delta, E, dos):
    mask = get_mask(epsilon, epsilon_k)
    return np.sum((dos[mask] / E[mask]) * (epsilon[mask] * alpha[mask] - Delta[mask] * nu[mask]) )

def compute_q_sum(alpha, nu, epsilon, Delta, E, g_unit, dos):
    return g_unit * np.array([
        single_q_sum(eps_k, alpha, nu, epsilon, Delta, E, dos) for eps_k in epsilon
    ])

def pair_creation_part(alpha, nu, epsilon, Delta, q_sum):
    return 2. * (epsilon * alpha - Delta * nu) + q_sum

def C_plus_minus(omega, alpha, nu, epsilon, Delta, q_sum):
    _pc = pair_creation_part(alpha, nu, epsilon, Delta, q_sum)
    return (omega * alpha + _pc, omega * alpha - _pc)

params = lattice_cut_params(N=N, 
                            g=G, 
                            U=0.0, 
                            E_F=E_F,
                            omega_D=OMEGA_D)

gap_df = load_panda("lattice_cut", f"{'bcs' if BCS else '.'}/{SYSTEM}", "gap.json.gz", **params, print_date=False)

redux_value = 0.04
redux_mask = np.abs(epsilon_full) < redux_value
epsilon = epsilon_full[redux_mask]

Delta = gap_df["Delta"][redux_mask]
dos = gap_df["dos"][redux_mask] * (epsilon[1] - epsilon[0])
E = np.sqrt(epsilon**2 + Delta**2)
g_unit = G_unit(G, dos, epsilon)
masks = [ get_mask(epsilon, k) for k in epsilon ]

main_df = load_panda("lattice_cut", f"{'bcs' if BCS else '.'}/{SYSTEM}", "full_diagonalization.json.gz", **params, print_date=False)
purger = fdp.FullDiagPurger(main_df, epsilon)



for PICK in range(min(len(purger.amplitude_eigenvalues), 4)):
    norm = np.max(np.abs(purger.amplitude_eigenvectors[PICK]))
    alpha = (purger.amplitude_eigenvectors[PICK][:N] / norm)[redux_mask]
    nu    = (purger.amplitude_eigenvectors[PICK][N:] / norm)[redux_mask]
    omega = purger.amplitude_eigenvalues[PICK]
    
    
    
    def right_side(beta):
        val = 4 * Delta**2 * beta - 4 * epsilon**2 * beta
        sums_single = 2 * g_unit * np.array([
            np.sum(dos[mask] * beta[mask] * (Delta[mask]**2 - epsilon[mask]**2) / E[mask]) for mask in masks
        ])
        sum_inner = g_unit * np.array([
            np.sum(dos[mask] * epsilon[mask] * beta[mask] / E[mask]) for mask in masks
        ])

        sums_single -= 2 * epsilon * sum_inner

        sum_double = -g_unit * np.array([
            np.sum(dos[mask] * epsilon[mask] * sum_inner[mask] / E[mask]) for mask in masks
        ])

        val += sums_single + sum_double
        return val
        
         
    q_sum = compute_q_sum(alpha, nu, epsilon, Delta, E, g_unit, dos)
    C_plus, C_minus = C_plus_minus(omega, alpha, nu, epsilon, Delta, q_sum)
    beta = (C_minus - C_plus)
    
    right_side_val = right_side(beta) / omega**2
    ax.plot(epsilon, (beta - right_side_val))
    
    for i in range(3):
        beta = right_side_val.copy()
        right_side_val = right_side(beta) / omega**2
        #if(i > 15):
        ax.plot(epsilon, (beta - right_side_val))

ax.set_xlim(-redux_value, redux_value)
#ax.set_ylim(-10, 10)
plt.show()