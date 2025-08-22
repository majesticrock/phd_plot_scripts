import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import __path_appender as __ap
__ap.append()
from legend import *
import string

XTYPE= "g"
X_LABEL = legend(r"\Delta_\mathrm{max}", "meV") if XTYPE == "Delta_max" else legend("g")
N_MODES = 5
N_SETS = 3

class Mode:
    def __init__(self, first_x, first_energy, first_weight, first_weight_error):
        self.x = [first_x]
        self.energies = [first_energy]
        self.weights = [first_weight]
        self.weight_errors = [first_weight_error]
        
    def append(self, new_x, new_energy, weight, weight_error):
        self.x.append(new_x)
        self.energies.append(new_energy)
        self.weights.append(weight)
        self.weight_errors.append(weight_error)
        
class ModeCollector:
    def __init__(self, first_x, first_energies, first_weights, first_weight_errors):
        self.modes = [ Mode(first_x, energy, weight, weight_error) for energy, weight, weight_error in zip(first_energies, first_weights, first_weight_errors) if energy is not None]
    
    def append_new_energy(self, new_x, new_energies, new_weights, new_weight_errors):
        EPS = 0.01
        for new_energy, weight, weight_error in zip(new_energies, new_weights, new_weight_errors):
            if new_energy is None:
                continue
            best_mode_idx = None
            best_energy_diff = 10000
            for j, mode in enumerate(self.modes):
                if np.abs(mode.energies[-1] - new_energy) < best_energy_diff:
                    best_energy_diff = np.abs(mode.energies[-1] - new_energy)
                    best_mode_idx = j
            if best_mode_idx is None or best_energy_diff > EPS:
                self.modes.append(Mode(new_x, new_energy, weight, weight_error))
            else:
                self.modes[best_mode_idx].append(new_x, new_energy, weight, weight_error)

lss = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1)), "-.", ":"]

fig, axes = plt.subplots(nrows=3, ncols=N_SETS, sharex="col", sharey="row", figsize=(12.8, 6))
axes = axes.transpose()
fig.subplots_adjust(hspace=0, wspace=0)

for n_mode in range(N_SETS):
    for j, ax in enumerate(axes[n_mode]):
        ax.text(0.015, 0.84, f"({string.ascii_lowercase[j]}.{1 + n_mode})", transform=ax.transAxes)
    
    (ax, ax_w_h, ax_w_p) = axes[n_mode]
    #ax_w_h.set_yscale("log")
    #ax_w_p.set_yscale("log")

    for idx, spectral_type in enumerate(["higgs", "phase"]):
        ax_w = ax_w_h if idx == 0 else ax_w_p
        df = pd.read_pickle(f"phd_plot_scripts/LatticeCUT/modes/{spectral_type}_{n_mode}.pkl").sort_values("g")
        filtered_df = df[df['energies'].apply(len) > 0].reset_index()

        for i, df_row in filtered_df.iterrows():
            if i == 0:
                modes = ModeCollector(df_row[XTYPE], df_row["energies"], df_row["weights"], df_row["weight_errors"])
            else:
                modes.append_new_energy(df_row[XTYPE], df_row["energies"], df_row["weights"], df_row["weight_errors"])

        ax.plot(df[XTYPE], df["true_gap"], "k-", zorder=-100)
        ls = 0
        for mode in modes.modes:
            if mode.energies[-1] < 1e-3:
                continue
            ax.plot(mode.x, mode.energies, color=f"C{idx}", ls=lss[ls%len(lss)], linewidth=(3.5 - 0.5 * idx))
            ax.plot(mode.x[0], mode.energies[0], color=f"C{idx}", marker="*", markersize=8, ls=None)
            ax_w.plot(mode.x, mode.weights, color=f"C{idx}", ls=lss[ls%len(lss)])
            ls += 1

axes[0][0].set_ylabel(legend(r"\omega_0", "meV"))
axes[0][1].set_ylabel(legend(r"W_\mathrm{H}"))
axes[0][2].set_ylabel(legend(r"W_\mathrm{P}"))

for i in range(3):
    axes[i][2].set_xlabel(X_LABEL)

legend_elements = [
    Line2D([0], [0], color='C0', linestyle=lss[0], label='Higgs'),
    Line2D([0], [0], color='C1', linestyle=lss[0], label='Phase')
]
axes[1][0].legend(handles=legend_elements, loc="upper center", 
                  borderaxespad=0.3, 
                  handlelength=1, 
                  handletextpad=0.3,
                  labelspacing=0.3,
                  borderpad=0.3,
                  bbox_to_anchor=(0.35, 1))
#axes[0][0].set_ylim(0, 55)
#fig.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.pdf")

plt.show()
