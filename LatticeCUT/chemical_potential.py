import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
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

def compute_normal_state_mu(epsilons, dos, beta, filling):
    from scipy.optimize import bisect
    def filling_diff(mu):
        return filling - np.sum(dos * fermi_function(epsilons - mu, beta)) * 2. / len(epsilons) # Delta epsilon = 2 / N
    return bisect(filling_diff, -1.0, 1.0, xtol=1e-10)


if __name__ == "__main__":
    N=10000
    SYSTEM = 'bcc'
    E_F=-0.5
    G=2.3
    U=0.01
    OMEGA_D=0.02
    params = lattice_cut_params(N=N, 
                                g=G,
                                U=U, 
                                E_F=E_F,
                                omega_D=OMEGA_D)
    tc_df   = load_panda("lattice_cut", f"./T_C/{SYSTEM}", "T_C.json.gz", **params)
    dos_df  = load_panda("lattice_cut", f"./old_bcc", "gap.json.gz",
                        **lattice_cut_params(N=N,  g=0., U=0.,  E_F=0., omega_D=0.02))

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$\mu - E_\mathrm{F}$')

    temps = tc_df['temperatures']
    chemical_potentials = np.array(tc_df['chemical_potentials']) - E_F

    ax.plot(temps, chemical_potentials, "-", label=f"SC")
    eps = np.linspace(-1, 1, N)
    chemical_potentials_ns = np.array([
            compute_normal_state_mu(eps, dos_df["dos"], 
                                    1. / np.where(T > 0, T, 1e-6), 
                                    tc_df["filling_at_zero_temp"])
                for T in temps
        ]) - E_F
    ax.plot(temps, chemical_potentials_ns, "--", label="Normal state")
    ax.legend()
    fig.tight_layout()
    plt.show()