import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *
from scipy.interpolate import CubicSpline
import numpy as np
from find_roots import *


N       = 16000
OMEGA_D = 0.02
G       = 1.5
U       = 0
E_F     = -0.5
SYSTEM  = 'fcc'

energies = np.linspace(-1, 1, N)

def compute_qp_dos(mu, gaps, sp_dos, max_E=None, plot_min=None):
    quasiparticle_energies = np.sqrt((energies - mu)**2 + gaps**2)
    true_gap = np.min(quasiparticle_energies)
    E_inter = CubicSpline(energies, quasiparticle_energies)
    E_deriv = E_inter.derivative()

    if max_E is None:
        max_E = 1.05 * np.max(quasiparticle_energies)
    if plot_min is None:
        plot_min = -(max(quasiparticle_energies[N - 1 - np.argmax(np.abs(gaps[::-1]) > 1e-6)], quasiparticle_energies[0]) + 0.05)

    omegas = np.linspace(true_gap + (1. / N), max_E, 2000)
    func = [ CubicSpline(energies, quasiparticle_energies - omega) for omega in omegas ]
    qp_dos_pos = np.zeros_like(func)
    qp_dos_neg = np.zeros_like(func)

    def integrand(epsilon_0):
        return sp_dos(epsilon_0) / np.where(np.abs(E_deriv(epsilon_0)) > 1e-12, np.abs(E_deriv(epsilon_0)), 1e-12)
    def u_k(eps):
        return 0.5 * (1 + np.where(E_inter(eps) > 1e-12, (eps - mu) / E_inter(eps), np.sign(eps - mu)))
    def v_k(eps):
        return 0.5 * (1 - np.where(E_inter(eps) > 1e-12, (eps - mu) / E_inter(eps), np.sign(eps - mu)))
    
    for i, g in enumerate(func):
        roots = find_all_roots_alt(g, energies)
        for root in roots:
            qp_dos_pos[i] += integrand(root) * u_k(root)
    for i, g in enumerate(func):
        roots = find_all_roots_alt(g, energies)
        for root in roots:
            qp_dos_neg[i] += integrand(root) * v_k(root)
            
    if true_gap > (10. / N):
        omegas_plot = np.concatenate([-omegas[::-1], np.array([-true_gap, true_gap]), omegas])
        qp_dos_plot = np.concatenate([qp_dos_neg[::-1], np.array([0.0, 0.0]), qp_dos_pos])
    else:
        omegas_plot = np.concatenate([-omegas[::-1], omegas])
        qp_dos_plot = np.concatenate([qp_dos_neg[::-1], qp_dos_pos])
    
    return omegas_plot, qp_dos_plot, max_E, plot_min

def create_qpdos_plot(ax, G, E_F, U, OMEGA_D, max_E=None):
    main_df = load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz",
                        **lattice_cut_params(N=N, 
                                             g=G,
                                             U=U, 
                                             E_F=E_F,
                                             omega_D=OMEGA_D))

    sp_dos = CubicSpline(energies, main_df['dos'])
    gaps = main_df['Delta']
    mu = main_df['chemical_potential']

    if max_E is None:
        omegas_plot, qp_dos_plot, max_E, plot_min = compute_qp_dos(mu, gaps, sp_dos)
    else:
        omegas_plot, qp_dos_plot, _, __ = compute_qp_dos(mu, gaps, sp_dos, max_E, -max_E)
        
    ax.plot(omegas_plot, qp_dos_plot)
    ax.set_xlabel(r"$E / W$")
    
    res_df = load_panda("lattice_cut", f"./{SYSTEM}", "resolvents.json.gz",
                        **lattice_cut_params(N=N, 
                                             g=G,
                                             U=U, 
                                             E_F=E_F,
                                             omega_D=OMEGA_D))
    ax.axvline(res_df["Delta_max"], c="k", ls="--")
    ax.axvline(res_df["continuum_boundaries"][0] * 0.5, c="k", ls=":")
    ax.axvline(-res_df["Delta_max"], c="k", ls="--")
    ax.axvline(-res_df["continuum_boundaries"][0] * 0.5, c="k", ls=":")
    

fig, ax = plt.subplots()
ax.set_ylabel(r"$\rho_\mathrm{qp}(E) / W^{-1}$")
create_qpdos_plot(ax, G, E_F, U, OMEGA_D, 0.5)
plt.show()