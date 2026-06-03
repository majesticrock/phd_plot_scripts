import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

from chemical_potential import *

def quasiparticle_dispersion(epsilons, deltas, E_F):
    return np.sqrt((epsilons - E_F)**2 + deltas**2)

def compute_normal_state_internal_energy(epsilons, dos, beta, mu):
    return np.sum(dos * (epsilons - mu) * fermi_function(epsilons - mu, beta)) * 2. / len(epsilons) # Delta epsilon = 2 / N

def compute_internal_energy(epsilons, delta, dos, beta, E_F):
    mask = np.abs(delta) > 0
    E = quasiparticle_dispersion(epsilons, delta, E_F)
    qp_part = 2 * np.sum(E[mask] * fermi_function(E[mask], beta) * dos[mask])
    single_particle_part = np.sum(dos[mask] * (epsilons[mask] - E_F - E[mask]))
    single_particle_part += 2 * np.sum(dos[~mask] * (epsilons[~mask] - E_F) * fermi_function(epsilons[~mask] - E_F, beta))
    if beta < 0:
        bcs_part = np.sum(0.5 * dos[mask] * delta[mask]**2 / E[mask])
    else:
        bcs_part = np.sum(0.5 * dos[mask] * delta[mask]**2 / E[mask] * np.tanh(0.5 * beta * E[mask]))
    total = qp_part + single_particle_part + bcs_part
    check = 2 * np.sum(dos * (epsilons - E_F) * fermi_function(epsilons - E_F, beta))
    if (total - check > 0 and beta < 0):
        print(total, check)
    return total / (0.5 * len(epsilons)) # Delta epsilon = 2 / N

def compute_heat_capacity(internal_energies, temperatures):
    return np.gradient(internal_energies, temperatures)

fig, ax = plt.subplots()
N=10000
SYSTEM = 'bcc'
E_F=-0.5
G=1.874
OMEGA_D=0.02
params = lattice_cut_params(N=N, 
                            g=G,
                            U=0., 
                            E_F=E_F,
                            omega_D=OMEGA_D)

tc_df   = load_panda("lattice_cut", f"./T_C/{SYSTEM}", "T_C.json.gz", **params)
main_df = load_panda("lattice_cut", f"./T_C/{SYSTEM}", "all_gaps.json.gz", **params)
dos_df  = load_panda("lattice_cut", f"./old_bcc", "gap.json.gz",
                    **lattice_cut_params(N=N,  g=0., U=0.,  E_F=0., omega_D=0.02))


Ts = tc_df["temperatures"]
deltas = main_df["finite_gaps"]
T_C = Ts[-1]
epsilons = np.linspace(-1, 1, N)
mus_sc = tc_df["chemical_potentials"]
internal_energies = np.array([
                compute_internal_energy(np.linspace(-1, 1, N), delta, 
                                        dos_df["dos"], 1. / np.where(T > 0, T, 1e-6), 
                                        mu)
                    for delta, T, mu in zip(deltas, Ts, mus_sc)
            ])
heat_capacities = compute_heat_capacity(internal_energies, Ts)
ax.plot(Ts / T_C, heat_capacities, label=f"g={main_df['g']:.2f}")

ax2 = ax.twinx()
ax2.plot(Ts/T_C, tc_df["max_gaps"], ls=":", c="k")

Ts_ns = np.concatenate([np.linspace(0.0, 0.99 * T_C, 100), np.linspace(T_C, 1.5 * T_C, 20)])
deltas_ns = np.zeros((len(Ts_ns), N))


mus_ns = np.array([
        compute_normal_state_mu(np.linspace(-1, 1, N), 
                                dos_df["dos"],
                                1. / np.where(T > 0, T, 1e-6),
                                tc_df["filling_at_zero_temp"])
            for T in Ts_ns
    ])
filling_at_zero_temp = tc_df["filling_at_zero_temp"]
internal_energies_ns = np.array([compute_normal_state_internal_energy(np.linspace(-1, 1, N),
                                                                      dos_df["dos"], 
                                                                      1. / np.where(T > 0, T, 1e-6), 
                                                                      mu)
                                                  for T, mu in zip(Ts_ns, mus_ns)])
heat_capacities_ns = compute_heat_capacity(internal_energies_ns, Ts_ns)
ax.plot(
    [Ts[-1] / T_C, Ts_ns[100] / T_C],
    [heat_capacities[-1], heat_capacities_ns[100]],
    ls="--"
)

ax.plot(Ts_ns / T_C, heat_capacities_ns, ls="--")
    

ax.set_xlabel(r"$T / T_c$")
ax.set_ylabel(r"$C_V$")

plt.show()