import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()

from get_data import *
fig, axes = plt.subplots(nrows=2, sharex=True)
fig.subplots_adjust(hspace=0)
SYSTEM = 'bcc'
G=2
U=0
E_F=0
OMEGA_D=0.02

for i, N in enumerate([16000, 32000]):#2000, 4000, 8000, 12000, 
    main_df = load_panda("lattice_cut", f"./{SYSTEM}", "resolvents.json.gz",
                        **lattice_cut_params(N=N, 
                                             g=G, 
                                             U=U, 
                                             E_F=E_F,
                                             omega_D=OMEGA_D))
    a_inf = (main_df["continuum_boundaries"][0]**2 + main_df["continuum_boundaries"][1]**2) * 0.5
    b_inf = (main_df["continuum_boundaries"][1]**2 - main_df["continuum_boundaries"][0]**2) * 0.25
    
    A = main_df["resolvents.amplitude_SC"][0]["a_i"]
    B = np.sqrt(main_df["resolvents.amplitude_SC"][0]["b_i"])
    axes[0].plot(A / a_inf, c=f"C{i}", label=f"$N={N}$")
    axes[0].plot(B / b_inf, c=f"C{i}", ls="--")
    
    A = main_df["resolvents.phase_SC"][0]["a_i"]
    B = np.sqrt(main_df["resolvents.phase_SC"][0]["b_i"])
    axes[1].plot(A / a_inf, c=f"C{i}", label=f"$N={N}$")
    axes[1].plot(B / b_inf, c=f"C{i}", ls="--")

for ax in axes:  
    ax.axhline(1, ls=":", c="k", label="$\infty$")
    ax.set_ylim(0.998, 1.002)
axes[0].set_ylabel("Higgs")
axes[1].set_ylabel("Phase")
ax.legend()
ax.set_xlabel("Iteration $i$")
fig.tight_layout()

plt.show()
