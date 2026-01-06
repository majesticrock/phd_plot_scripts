import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from get_data import *

SYSTEM = 'sc'
main_df = load_panda("lattice_cut", f"./test/{SYSTEM}", "residuals.json.gz",
                    **lattice_cut_params(N=4000, 
                                         g=2.5, 
                                         U=0.0, 
                                         E_F=0,
                                         omega_D=0.02), print_date=False)

fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True)
fig.subplots_adjust(hspace=0)
N = len(main_df["amplitude.eigenvectors"][1])

epsilon = np.linspace(-1, 1, N)

axes[0, 0].plot(epsilon, main_df["amplitude.eigenvectors"][1], label="$j=1$")
axes[0, 0].plot(epsilon, main_df["amplitude.eigenvectors"][3], label="$j=2$")
axes[0, 0].plot(epsilon, main_df["amplitude.eigenvectors"][5], label="$j=3$")

axes[1, 0].plot(epsilon, main_df["phase.eigenvectors"][1][:N])
axes[1, 0].plot(epsilon, main_df["phase.eigenvectors"][4][:N])

axes[2, 0].plot(epsilon, main_df["phase.eigenvectors"][1][N:])
axes[2, 0].plot(epsilon, main_df["phase.eigenvectors"][4][N:])

axes[0, 1].plot(epsilon, main_df["amplitude.transformed_vectors"][1][:N], label="1st mode")
axes[0, 1].plot(epsilon, main_df["amplitude.transformed_vectors"][3][:N], label="2nd mode")
axes[0, 1].plot(epsilon, main_df["amplitude.transformed_vectors"][5][:N], label="3rd mode")

axes[1, 1].plot(epsilon, main_df["amplitude.transformed_vectors"][1][N:], label="1st mode")
axes[1, 1].plot(epsilon, main_df["amplitude.transformed_vectors"][3][N:], label="2nd mode")
axes[1, 1].plot(epsilon, main_df["amplitude.transformed_vectors"][5][N:], label="3rd mode")

axes[2, 1].plot(epsilon, main_df["phase.transformed_vectors"][1])
axes[2, 1].plot(epsilon, main_df["phase.transformed_vectors"][4])


axes[0, 0].set_ylabel(r"$c^\dagger c^\dagger - c c$")
axes[1, 0].set_ylabel(r"$c^\dagger c^\dagger + c c$")
axes[2, 0].set_ylabel(r"$c^\dagger c$")

axes[0, 1].set_ylabel(r"$c^\dagger c^\dagger + c c$")
axes[1, 1].set_ylabel(r"$c^\dagger c$")
axes[2, 1].set_ylabel(r"$c^\dagger c^\dagger - c c$")

axes[0, 0].set_title(r"$| v_j \rangle$")
axes[0, 1].set_title(r"$|\psi_j \rangle $")

axes[-1, 0].set_xlabel(r"$\varepsilon / W$")
axes[-1, 1].set_xlabel(r"$\varepsilon / W$")
axes[0, 0].legend(loc="upper right")

plt.show()