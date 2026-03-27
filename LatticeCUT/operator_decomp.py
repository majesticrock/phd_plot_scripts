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

def single_q_sum(epsilon_k, alpha, nu, epsilon, Delta, E, dos):
    mask = np.abs(epsilon - epsilon_k) <= OMEGA_D_UNIT
    return np.sum(dos[mask] * (epsilon[mask] * alpha[mask] - Delta[mask] * nu[mask]) / E[mask])

def compute_q_sums(alpha, nu, epsilon, Delta, E, g, dos):
    _G = G_unit(g, dos, epsilon)
    return _G * np.array([
        single_q_sum(eps_k, alpha, nu, epsilon, Delta, E, dos) for eps_k in epsilon
    ])

for axes, axes_q, G in zip(axes_2d, axes_2d_q, [0.3, 3.0]):
    axes[0].set_ylabel(f"$g={G}$\n$\\alpha_j^{{(n)}}$")
    axes_q[0].set_ylabel(f"$g={G}$\n$q$-sums")
    for ax, ax_q, SYSTEM in zip(axes, axes_q, ["sc", "bcc", "fcc"]):
        if G==0.3:
            ax.set_title(SYSTEM)
            ax_q.set_title(SYSTEM)
        params = lattice_cut_params(N=N, 
                                    g=G, 
                                    U=0.0, 
                                    E_F=E_F,
                                    omega_D=OMEGA_D)
        main_df = load_panda("lattice_cut", f"./{SYSTEM}", "full_diagonalization.json.gz", **params, print_date=False)
        purger = fdp.FullDiagPurger(main_df, epsilon)

        gap_df = load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz", **params, print_date=False)
        Delta = gap_df["Delta"]
        dos = gap_df["dos"]
        E = np.sqrt(epsilon**2 + Delta**2)

        for PICK in range(min(len(purger.amplitude_eigenvalues), 4)):
            alpha = purger.amplitude_eigenvectors[PICK][:N]
            norm = np.max(np.abs(alpha))
            nu = purger.amplitude_eigenvectors[PICK][N:]

            q_sum = compute_q_sums(alpha, nu, epsilon, Delta, E, G, dos)
            
            f_plus = alpha * (purger.amplitude_eigenvalues[PICK] + 2 * epsilon) - 2 * Delta * nu + q_sum
            f_minus = alpha * (purger.amplitude_eigenvalues[PICK] - 2 * epsilon) + 2 * Delta * nu - q_sum
            
            ax_q.plot(epsilon, q_sum, color=colors[PICK], label=PICK)
            ax.plot(epsilon, f_plus, color=colors[PICK], label=PICK)
            ax.plot(epsilon[::-1], f_minus, color=colors[PICK], dashes=(3.5, 3.5), linewidth=4)


axes_2d[-1,-1].legend(loc="lower right")
axes_2d[0,0].set_xlim(-0.05, 0.05)
axes_2d[-1,0].set_xlim(-0.25, 0.25)

axes_2d_q[-1,-1].legend(loc="lower right")
axes_2d_q[0,0].set_xlim(-0.05, 0.05)
axes_2d_q[-1,0].set_xlim(-0.25, 0.25)

plt.show()