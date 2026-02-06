import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

def fermi_function(x, beta):
    if np.isscalar(beta) and beta < 0:
        return np.where(x < 0, 1., np.where(x > 0, 0., 0.5))
    __CUT__ = 38 # e^38 ~ 3.2e16
    arg = beta * x
    result = np.empty_like(arg, dtype=float)
    result[arg > __CUT__] = 0.0
    result[arg < -__CUT__] = 1.0
    mask = (arg >= -__CUT__) & (arg <= __CUT__)
    result[mask] = 1. / (1 + np.exp(arg[mask]))
    
    return result

def quasiparticle_dispersion(epsilons, deltas, E_F):
    return np.sqrt((epsilons - E_F)**2 + deltas**2)

def compute_internal_energy(epsilons, deltas, dos, beta, E_F):
    mask = np.abs(deltas) > 0
    E = quasiparticle_dispersion(epsilons, deltas, E_F)
    # number part = 2 * sum_k E_k
    qp_part = 2 * np.sum(E[mask] * fermi_function(E[mask], beta) * dos[mask])
    # single-particle part = sum_k epsilon_k - E_k
    single_particle_part = np.sum(dos[mask] * (epsilons[mask] - E_F - E[mask]))
    single_particle_part += 2 * np.sum(dos[~mask] * (epsilons[~mask] - E_F) * fermi_function(epsilons[~mask] - E_F, beta))
    # BCS-part
    if beta < 0:
        bcs_part = np.sum(0.5 * dos[mask] * deltas[mask]**2 / E[mask])
    else:
        bcs_part = np.sum(0.5 * dos[mask] * deltas[mask]**2 / E[mask] * np.tanh(0.5 * beta * E[mask]))
    
    total = qp_part + single_particle_part + bcs_part
    check = 2 * np.sum(dos * (epsilons - E_F) * fermi_function(epsilons - E_F, beta))
    if (total - check > 0 and beta < 0):
        #raise ValueError(f"Violated energy; should be negative but is not: {total - check}")
        print(total, check)
    return total / (0.5 * len(epsilons))

def compute_heat_capacity(internal_energies, temperatures):
    return np.gradient(internal_energies, temperatures)

fig, ax = plt.subplots()
N=10000
E_F=-0.5
U=0.0
OMEGA_D=0.02
SYSTEM = 'bcc'
main_df = load_all(f"lattice_cut/./T_C/{SYSTEM}/N={N}/", "all_gaps.json.gz", condition=[f"U={U}", f"E_F={E_F}", f"omega_D={OMEGA_D}"]).sort_values('g')
tc_df   = load_all(f"lattice_cut/./T_C/{SYSTEM}/N={N}/", "T_C.json.gz", condition=[f"U={U}", f"E_F={E_F}", f"omega_D={OMEGA_D}"]).sort_values('g')
dos_df  = load_panda("lattice_cut", f"./{SYSTEM}", "gap.json.gz",
                    **lattice_cut_params(N=N,  g=0., U=0.,  E_F=0., omega_D=0.02))

from matplotlib import cm
g_values = main_df['g'].to_numpy()
norm = plt.Normalize(vmin=np.min(g_values), vmax=np.max(g_values))
cmap = cm.gist_rainbow

for (i, row), (_, row_tc) in zip(main_df.iterrows(), tc_df.iterrows()):
    Ts = row_tc["temperatures"]
    deltas = row["finite_gaps"]
    T_C = Ts[-1]
    
    if (len(Ts) < 5):
        continue
    internal_energies = np.array([
                              compute_internal_energy(np.linspace(-1, 1, N), delta, dos_df["dos"], np.where(T > 0, 1. / T, 1e6), E_F) for delta, T in zip(deltas, Ts)
                              ])
    #ax.plot(Ts / T_C, internal_energies, color=cmap(norm(row['g'])), label=f"g={row['g']:.2f}")
    heat_capacities = compute_heat_capacity(internal_energies, Ts)
    ax.plot(Ts / T_C, heat_capacities, color=cmap(norm(row['g'])), label=f"g={row['g']:.2f}")

    Ts_ns = np.linspace(T_C, 1.5 * T_C, 20)
    deltas_ns = np.zeros((len(Ts_ns), N))
    internal_energies_ns = np.array([
                              compute_internal_energy(np.linspace(-1, 1, N), delta, dos_df["dos"], np.where(T > 0, 1. / T, 1e6), E_F) for delta, T in zip(deltas_ns, Ts_ns)
                              ])

    heat_capacities_ns = compute_heat_capacity(internal_energies_ns, Ts_ns)
    ax.plot(
        [Ts[-1] / T_C, Ts_ns[0] / T_C],
        [heat_capacities[-1], heat_capacities_ns[0]],
        color=cmap(norm(row['g'])),
        ls="--"
    )
    ax.plot(Ts_ns / T_C, heat_capacities_ns, color=cmap(norm(row['g'])), ls="--")
    
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label="$g$")

ax.set_xlabel(r"$T / T_c$")
ax.set_ylabel(r"$C_V$")

plt.show()