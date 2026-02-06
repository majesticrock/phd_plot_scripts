import matplotlib.pyplot as plt
import numpy as np
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *

fig, axes = plt.subplots(nrows=2, sharex=True)

labels = [
    "sc",
    "bcc",
    "fcc"
]
datas = [
    load_all("lattice_cut/sc/N=16000",  "gap.json.gz"),
    load_all("lattice_cut/bcc/N=16000", "gap.json.gz"),
    load_all("lattice_cut/fcc/N=16000", "gap.json.gz")
]

for i, (data, label) in enumerate(zip(datas, labels)):
    query = data.query(f"omega_D == 0.02 & g >= 0.2 & g <= 2.5 & Delta_max > 0 & E_F == 0", engine="python").sort_values("g")
    #y_data = np.array([0.5 * boundaries[0] for boundaries in query["continuum_boundaries"]])
    argmax_deltas = np.array( [ np.argmax(delta) for  delta in query["Delta"] ] )
    
    second_derivative = np.array( [ (delta[arg - 1] - 2 * delta[arg] + delta[arg + 1]) / ((energies[arg + 1] - energies[arg])**2) for arg, delta, energies in zip(argmax_deltas, query["Delta"], query["energies"]) ] )
    second_order = np.array( [ 1 + delta[arg] * second 
                              for arg, second, delta in zip(argmax_deltas, second_derivative, query["Delta"]) ] )
    axes[0].plot(query["g"], second_order, label=label, color=f"C{i}")
    axes[0].axhline(0, color="black", linestyle=":")
    
    fourth_derivative = np.array( [ (delta[arg - 2] - 4 * delta[arg - 1] + 6 * delta[arg] - 4 * delta[arg + 1] + delta[arg + 2]) / ((energies[arg + 1] - energies[arg])**4) for arg, delta, energies in zip(argmax_deltas, query["Delta"], query["energies"]) ] )
    fourth_order = np.array( [ delta[arg]**3 * fourth - 6 * delta[arg] * second - 3
                              for arg, second, fourth, delta in zip(argmax_deltas, second_derivative, fourth_derivative, query["Delta"]) ] )
    axes[1].plot(query["g"], fourth_order, label=label, color=f"C{i}")
    axes[1].axhline(0, color="black", linestyle=":")
    
axes[0].set_ylabel(r"Second order coefficient")
axes[1].set_ylabel(r"Fourth order coefficient")
axes[1].set_xlabel(r"$g$")
axes[0].legend()

plt.show()