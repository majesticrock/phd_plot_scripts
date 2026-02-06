import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *
from scipy.optimize import minimize_scalar
import matplotlib.colors as cm

__LAST_MU__ = 0.0

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

def compute_normal_state_mu(epsilons, dos, beta, filling, guess=0.0, guess_tol=0.5):
    def filling_diff(mu):
        return (filling - np.sum(dos * fermi_function(epsilons - mu, beta)) * 2. / len(epsilons))**2 # Delta epsilon = 2 / N
    res = minimize_scalar(filling_diff, method='bounded', bounds=(guess - guess_tol, guess + guess_tol))
    if not res.success:
        print("Warning: minimization for mu did not converge.")
    __LAST_MU__ = res.x
    return __LAST_MU__


if __name__ == "__main__":
    N=10000
    SYSTEM = 'bcc'
    E_F=-0.5
    U=0.2
    OMEGA_D=0.02
    
    G_ENH = 1.88 if U==0 else 1.95
    
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$T / W$')
    ax.set_ylabel(r'$(\mu - E_\mathrm{F}) / W$')
    
    Gs = np.array([1.9, 1.925, 1.95, 1.975, 2.0, 2.025, 2.05, 2.075])
    cmap_below = plt.get_cmap("YlGnBu")
    cmap_above = plt.get_cmap("hot_r")
    norm_below = cm.Normalize(Gs[0] - 0.4, G_ENH)
    norm_above = cm.Normalize(G_ENH - 0.2, Gs[-1] + 0.05)
    
    dos_df  = load_panda("lattice_cut", f"./old_bcc", "gap.json.gz",
                            **lattice_cut_params(N=N,  g=0., U=0.,  E_F=0., omega_D=0.02))
    
    for i, G in enumerate(Gs):
        params = lattice_cut_params(N=N, 
                                    g=G,
                                    U=U, 
                                    E_F=E_F,
                                    omega_D=OMEGA_D)
        tc_df   = load_panda("lattice_cut", f"./T_C/{SYSTEM}", "T_C.json.gz", **params)
        temps = tc_df['temperatures']
        chemical_potentials = np.array(tc_df['chemical_potentials']) - E_F

        ls = "-" if G > G_ENH else "--"
        color = cmap_below(norm_below(G)) if G < G_ENH else cmap_above(norm_above(G))
        ax.plot(temps, chemical_potentials, c=color, label=f"${G}$", ls=ls)
        
    eps = np.linspace(-1, 1, N)
    __LAST_MU__ = E_F
    chemical_potentials_ns = np.array([
            compute_normal_state_mu(eps, dos_df["dos"], 
                                    1. / np.where(T > 0, T, 1e-6), 
                                    tc_df["filling_at_zero_temp"],
                                    guess=__LAST_MU__)
                for T in temps
        ]) - E_F
    ax.plot(temps, chemical_potentials_ns, c="k", ls=":")#, label="Normal state")
    
    ax.legend(loc="upper right", ncols=2, title="$g$")
    
    #lim = -0.059
    #ax.set_ylim(lim, 0.001)
    #i = np.argmin(np.abs(lim - chemical_potentials_ns))
    #ax.set_xlim(0, temps[i])
    
    fig.tight_layout()
    plt.show()