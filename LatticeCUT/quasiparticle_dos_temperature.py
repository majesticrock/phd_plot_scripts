import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
__ap.append()
from get_data import *
from scipy.interpolate import interp1d
import numpy as np
from find_roots import find_all_roots

fig, ax = plt.subplots()

SYSTEM = 'bcc'
G=1.5
E_F=-0.5
U=0.
N=10000
main_df = load_panda("lattice_cut", f"T_C/{SYSTEM}", "all_gaps.json.gz",
                    **lattice_cut_params(N=N, 
                                         g=G,
                                         U=U, 
                                         E_F=E_F,
                                         omega_D=0.02))
tc_df   = load_panda("lattice_cut", f"./T_C/{SYSTEM}", "T_C.json.gz",
                    **lattice_cut_params(N=N, 
                                         g=G,
                                         U=U, 
                                         E_F=E_F,
                                         omega_D=0.02))
dos_df  = load_panda("lattice_cut", f"./old_bcc", "gap.json.gz",
                    **lattice_cut_params(N=N, 
                                         g=0.,
                                         U=0., 
                                         E_F=0.,
                                         omega_D=0.02))

energies = np.linspace(-1, 1, N)
sp_dos = interp1d(energies, dos_df['dos'], bounds_error=False, fill_value=0.0)

temps = tc_df['temperatures']
temps_picks = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.])
plot_indizes = np.argmin(np.abs(np.array(temps)[:, None] - temps_picks[None, :] * temps[-1]), axis=0)

n_curves = len(plot_indizes)
cmap = plt.get_cmap("viridis")

min_E = 0
max_E = 0

for c, idx in enumerate(plot_indizes):
    gaps = main_df['finite_gaps'][idx]
    mu = tc_df['chemical_potentials'][idx]
    
    quasiparticle_energies = np.sqrt((energies - mu)**2 + gaps**2)
    true_gap = tc_df["true_gaps"][idx]
    true_gap_at_eps_to_EF = abs(mu - energies[np.argmin(quasiparticle_energies)])
    E_inter = interp1d(energies, quasiparticle_energies, bounds_error=False, fill_value=0.0)
    E_deriv_inter = interp1d(energies, np.gradient(quasiparticle_energies, energies[1] - energies[0]), bounds_error=False, fill_value=0.0)

    omegas = np.linspace(1.1 * np.max(quasiparticle_energies), 0.0, 200, endpoint=False)[::-1]
    func = [ interp1d(energies, quasiparticle_energies - omega, bounds_error=False, fill_value=0.0) for omega in omegas ]

    from scipy.optimize import brentq
    qp_dos_pos = np.zeros_like(func)
    qp_dos_neg = np.zeros_like(func)

    def integrand(epsilon_0):
        return sp_dos(epsilon_0) / np.where(np.abs(E_deriv_inter(epsilon_0)) > 1e-12, np.abs(E_deriv_inter(epsilon_0)), 1e-12)
    def u_k(eps):
        return 0.5 * (1 + np.where(E_inter(eps) > 1e-12, (eps - mu) / E_inter(eps), np.sign(eps - mu)))
    def v_k(eps):
        return 0.5 * (1 - np.where(E_inter(eps) > 1e-12, (eps - mu) / E_inter(eps), np.sign(eps - mu)))

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
        roots = find_all_roots(g)
        for root in roots:
            qp_dos_pos[i] += integrand(root) * u_k(root)
        
    for i, g in enumerate(func):
        roots = find_all_roots(g)
        for root in roots:
            qp_dos_neg[i] += integrand(root) * v_k(root)

    color = cmap(c / (n_curves - 1) if n_curves > 1 else 0.0)
    
    
    none_zero_dos_index = len(qp_dos_pos) - 1 - np.argmax(qp_dos_neg[::-1] > 1e-8)
    if(-omegas[none_zero_dos_index] < min_E):
        min_E = -omegas[none_zero_dos_index]
    none_zero_dos_index = len(qp_dos_pos) - 1 - np.argmax(qp_dos_pos[::-1] > 1e-8)
    if(omegas[none_zero_dos_index] > max_E):
        max_E = omegas[none_zero_dos_index]
    
    AX_DISTANCE = 0.5
    ax.plot(np.concatenate([-omegas[::-1], omegas]), AX_DISTANCE * c + np.concatenate([qp_dos_neg[::-1], qp_dos_pos]),
            color=color, label=f"$T={temps_picks[c]}T_c$")
    if c != 0:
        axh = ax.secondary_xaxis(AX_DISTANCE * c, transform=ax.transData)
        axh.set_xticklabels([])
        axh.tick_params(axis='x', direction='inout', which='both')
        axh.tick_params(axis='x', length=8)          # major ticks
        axh.tick_params(axis='x', which='minor', length=5)

ax.set_xlim(min_E - 0.05, max_E + 0.05)
ax.legend()
ax.set_xlabel(r"$E / W$")
ax.set_ylabel(r"$\rho_\mathrm{qp}(E) / W^{-1}$")
ax.set_ylim(0, 5.5)

plt.show()