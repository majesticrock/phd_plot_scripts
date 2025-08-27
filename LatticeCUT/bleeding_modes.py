import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from get_data import *
import pickle

mode = pickle.load(open("phd_plot_scripts/LatticeCUT/modes/phase_fcc_first_secondary.pkl", "rb"))

fig, ax = plt.subplots()
weights = 1e4 * np.array(mode.weights)
errors  = 1e4 * np.array(mode.weight_errors) 

ax.plot(mode.x, weights, "o-", label="first bleeding mode")

ph_ness = np.zeros(len(mode.x))
for i, g in enumerate(mode.x):
    main_df = load_panda('lattice_cut', f'./fcc', 'gap.json.gz',
                    **lattice_cut_params(N=16000, 
                                         g=g, 
                                         U=0, 
                                         E_F=0,
                                         omega_D=0.02), print_date=False)
    ph_ness[i] = np.sum(main_df['energies'] * main_df['Delta'])

ax2 = ax.twinx()
ax2.plot(mode.x, np.abs(ph_ness), "x--", label="particle_holeness", color="red")
ax2.set_ylabel("Violation of PH", color="red")
ax2.tick_params(axis='y', colors='red')
ax2.yaxis.label.set_color('red')

for __n, marker in zip(["second", "third"], ["^", "v"]):
    mode = pickle.load(open(f"phd_plot_scripts/LatticeCUT/modes/phase_fcc_{__n}_secondary.pkl", "rb"))
    
    weights = 1e4 * np.array(mode.weights)
    errors  = 1e4 * np.array(mode.weight_errors) 
    ax.plot(mode.x, weights, f"{marker}-", label=f"{__n} bleeding mode")


ax.set_ylabel("Weight $\\times 10^4$", color="C0")
ax.tick_params(axis='y', colors='C0')
ax.yaxis.label.set_color('C0')

ax.legend(loc="upper left")
ax.set_xlabel("g")

plt.show()