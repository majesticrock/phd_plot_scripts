import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *
from scipy.interpolate import interp1d

fig, ax = plt.subplots()

SYSTEM = 'bcc'
U=0
E_F=-0.5
dfs = [
    load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz",
                    **lattice_cut_params(N=16000, 
                                         g=1.0,
                                         U=U, 
                                         E_F=E_F,
                                         omega_D=0.02)),
    load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz",
                    **lattice_cut_params(N=16000, 
                                         g=2.0,
                                         U=U, 
                                         E_F=E_F,
                                         omega_D=0.02)),
    load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz",
                    **lattice_cut_params(N=16000, 
                                         g=2.5,
                                         U=U, 
                                         E_F=E_F,
                                         omega_D=0.02)),
    ]

for main_df in dfs:
    G = main_df['g']
    
    energies = main_df['energies']
    sp_dos = interp1d(main_df['energies'], main_df['dos'], bounds_error=False, fill_value=0.0)
    gaps = main_df['Delta']
    gaps_inter = interp1d(energies, gaps, bounds_error=False, fill_value=0.0)

    quasiparticle_energies = np.sqrt((energies - E_F)**2 + gaps**2)
    true_gap = np.min(quasiparticle_energies)
    true_gap_at_eps_to_EF = abs(E_F - energies[np.argmin(quasiparticle_energies)])

    E_inter = interp1d(energies, quasiparticle_energies, bounds_error=False, fill_value=0.0)
    E_deriv_inter = interp1d(energies, np.gradient(quasiparticle_energies, energies[1] - energies[0]), bounds_error=False, fill_value=0.0)

    print(true_gap)

    omegas = np.linspace(0.0, 1.05 * np.max(quasiparticle_energies), 2000)
    func = [ interp1d(energies, quasiparticle_energies - omega, bounds_error=False, fill_value=0.0) for omega in omegas ]

    from scipy.optimize import brentq
    qp_dos_pos = np.zeros_like(func)
    qp_dos_neg = np.zeros_like(func)

    def integrand(epsilon_0):
        return sp_dos(epsilon_0) / np.abs(E_deriv_inter(epsilon_0))
    def u_k(eps):
        return 0.5 * (1 + (eps - E_F) / E_inter(eps))
    def v_k(eps):
        return 0.5 * (1 - (eps - E_F) / E_inter(eps))

    def u_integrand_if_root_exists(g, a, b):
        if abs(a-b) < 1e-12:
            return 0.0
        try:
            epsilon_0 = brentq(g, a, b)
            return integrand(epsilon_0) * u_k(epsilon_0)
        except ValueError:
            return 0.0

    def v_integrand_if_root_exists(g, a, b):
        if abs(a-b) < 1e-12:
            return 0.0
        try:
            epsilon_0 = brentq(g, a, b)
            return integrand(epsilon_0) * v_k(epsilon_0)
        except ValueError:
            return 0.0

    for i, g in enumerate(func):
        qp_dos_pos[i] += u_integrand_if_root_exists(g, -1., -abs(E_F)-true_gap_at_eps_to_EF) 
        qp_dos_pos[i] += u_integrand_if_root_exists(g, -abs(E_F)-true_gap_at_eps_to_EF, -abs(E_F))
        qp_dos_pos[i] += u_integrand_if_root_exists(g, -abs(E_F), abs(E_F)+true_gap_at_eps_to_EF)
        qp_dos_pos[i] += u_integrand_if_root_exists(g, abs(E_F)+true_gap_at_eps_to_EF, 1.)

    for i, g in enumerate(func):
        qp_dos_neg[i] += v_integrand_if_root_exists(g, -1., -abs(E_F)-true_gap_at_eps_to_EF) 
        qp_dos_neg[i] += v_integrand_if_root_exists(g, -abs(E_F)-true_gap_at_eps_to_EF, -abs(E_F))
        qp_dos_neg[i] += v_integrand_if_root_exists(g, -abs(E_F), abs(E_F)+true_gap_at_eps_to_EF)
        qp_dos_neg[i] += v_integrand_if_root_exists(g, abs(E_F)+true_gap_at_eps_to_EF, 1.)

    ax.plot(np.concatenate([-omegas[::-1], omegas]), np.concatenate([qp_dos_neg[::-1], qp_dos_pos]),
            label=f"$E_F = {E_F}, g={G}$")

ax.legend()
ax.set_xlabel(r"$E / W$")
ax.set_ylabel(r"$\rho_\mathrm{qp}(E) / W^{-1}$")
ax.set_ylim(0, 4)
fig.tight_layout()

plt.show()