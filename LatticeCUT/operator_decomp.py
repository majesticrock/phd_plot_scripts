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
epsilon = np.linspace(-1, 1, N) - E_F

BCS = True

colors = [
    "#000000",
    "#F97316",
    "#5105D3",
    "#5DDFFF",
]

fig, axes_2d = create_large_figure(ncols=3, nrows=2, sharey="row", sharex="row", height_to_width_ratio=3.6/6.4)
fig.subplots_adjust(wspace=0) 
for ax in axes_2d.ravel():
    ax.set_xlabel(r"$\varepsilon / W$")

fig_q, axes_2d_q = create_large_figure(ncols=3, nrows=2, sharey="row", sharex="row", height_to_width_ratio=3.6/6.4)
fig_q.subplots_adjust(wspace=0) 
for ax in axes_2d_q.ravel():
    ax.set_xlabel(r"$\varepsilon / W$")

epsilon = np.linspace(-1, 1, N) - E_F

def G_unit(g, dos, epsilon):
    rho_tilde = np.sum(dos[np.abs(epsilon <= OMEGA_D_UNIT)]) / (2 * OMEGA_D_UNIT)
    return g / (2 * rho_tilde)

def get_mask(epsilon, epsilon_k):
    if BCS:
        return np.abs(epsilon) <= OMEGA_D_UNIT
    else:
        return np.abs(epsilon - epsilon_k) <= OMEGA_D_UNIT 

def g_sum(epsilon_k, integrand):
    mask = get_mask(epsilon, epsilon_k)
    return np.sum(dos[mask] * integrand[mask])

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

for axes, axes_q, G in zip(axes_2d, axes_2d_q, [0.3]):#, 3.0
    axes[0].set_ylabel(f"$g={G}$\n$\\alpha_j^{{(n)}}$")
    axes_q[0].set_ylabel(f"$g={G}$\n$q$-sums")
    for ax, ax_q, SYSTEM in zip(axes, axes_q, ["sc"]):#, "bcc", "fcc"
        if G==0.3:
            ax.set_title(SYSTEM)
            ax_q.set_title(SYSTEM)
        params = lattice_cut_params(N=N, 
                                    g=G, 
                                    U=0.0, 
                                    E_F=E_F,
                                    omega_D=OMEGA_D)
        main_df = load_panda("lattice_cut", f"{'bcs' if BCS else '.'}/{SYSTEM}", "full_diagonalization.json.gz", **params, print_date=False)
        purger = fdp.FullDiagPurger(main_df, epsilon)

        gap_df = load_panda("lattice_cut", f"{'bcs' if BCS else '.'}/{SYSTEM}", "gap.json.gz", **params, print_date=False)
        Delta = gap_df["Delta"]
        dos = gap_df["dos"]
        E = np.sqrt(epsilon**2 + Delta**2)

        g_unit = G_unit(G, dos, epsilon)
        for PICK in range(min(len(purger.amplitude_eigenvalues), 4)):
            norm = np.max(np.abs(purger.amplitude_eigenvectors[PICK]))
            alpha = purger.amplitude_eigenvectors[PICK][:N] / norm
            nu = purger.amplitude_eigenvectors[PICK][N:] / norm
            omega = purger.amplitude_eigenvalues[PICK]
            
            q_sum = compute_q_sum(alpha, nu, epsilon, Delta, E, g_unit, dos)
            C_plus, C_minus = C_plus_minus(omega, alpha, nu, epsilon, Delta, q_sum)
            
            #ax_q.plot(epsilon, q_sum, color=colors[PICK], label=PICK)
            #ax.plot(epsilon, C_plus, color=colors[PICK], label=PICK)
            #ax.plot(epsilon[::-1], C_minus, color=colors[PICK], dashes=(3.5, 3.5), linewidth=4)
            
            check_n = (nu - 2 * Delta * (-2*alpha*epsilon + 2*Delta*nu - 2*q_sum) / omega**2)
            
            #ax_q.plot(epsilon, nu, color=colors[PICK+1])
            #ax_q.plot(epsilon, 2 * Delta * (-2*alpha*epsilon + 2*Delta*nu)  / omega**2 , color=colors[PICK+2], dashes=(4,4))
            #ax_q.plot(epsilon, -4 * Delta * q_sum / omega**2 , color=colors[PICK+3])
            ax_q.plot(epsilon, check_n , color=colors[PICK], ls="-.")
            
            C_plus_sum = g_unit * np.array([
                g_sum(k, epsilon * C_plus / E) for k in epsilon
            ])
            nu_sum = g_unit * np.array([
                g_sum(k, Delta * nu / E) for k in epsilon
            ])
            
            check_plus = (C_plus - (2 * epsilon * C_plus + C_plus_sum) / omega + (2 * Delta * nu + nu_sum)) / omega
            ax.plot(epsilon, check_plus, color=colors[PICK])
            
            C_minus_sum = g_unit * np.array([
                g_sum(k, epsilon * C_minus / E) for k in epsilon
            ])
            check_minus = (C_minus + (2 * epsilon * C_minus + C_minus_sum) / omega - (2 * Delta * nu + nu_sum)) / omega
            ax.plot(epsilon, check_minus, color=colors[PICK], dashes=(4,4))

axes_2d[-1,-1].legend(loc="lower right")
axes_2d[0,0].set_xlim(-0.05, 0.05)
axes_2d[-1,0].set_xlim(-0.25, 0.25)

axes_2d_q[-1,-1].legend(loc="lower right")
axes_2d_q[0,0].set_xlim(-0.05, 0.05)
axes_2d_q[-1,0].set_xlim(-0.25, 0.25)

plt.show()