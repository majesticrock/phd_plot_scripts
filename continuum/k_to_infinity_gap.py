from turtle import pd
import matplotlib.pyplot as plt
import numpy as np

from mrock.get_data import *
data_loader = DataLoader()

MEV_FACTOR = 1e3

fig, ax = plt.subplots()

main_df = data_loader.load_panda("continuum", "offset_10", "gap.json.gz", 
                                 **continuum_params(N_k=20000,
                                                    T=0,
                                                    g=1, 
                                                    coulomb_scaling=1, 
                                                    screening=1e-4,
                                                    k_F=4.25, 
                                                    omega_D=10.))
pd_data = main_df["data"]
pd_data["total"] = np.abs( pd_data["Delta_Coulomb"])
pd_data.plot("ks", "total", ax=ax, label=r"$\Delta_\mathrm{Coulomb}$")

x_lin = np.linspace(0.1 * pd_data["ks"].max(), 2. * pd_data["ks"].max(), 200)
y_inf = 1e3 * np.abs(main_df["k_infinity_factor"]) / x_lin**2
y_zero = 1e3 * np.abs(main_df["k_zero_factor"])

ax.plot(x_lin, y_inf, "--", c="C1", label=r"$k\to\infty$")
ax.axhline(y_zero, ls="--", c="C3", label=r"$k\to 0$")

ax.set_xlabel(r"$k [\sqrt{\mathrm{eV}}]$")
ax.set_ylabel(r"$|\Delta|[\mathrm{meV}]$")

ax.set_yscale("log")
ax.set_xscale("log")

ax.legend()
fig.tight_layout()

plt.show()