import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
import get_data
from create_figure import *

N=10000
U=0.0
E_F=-0.5
omega_D=0.02
main_df = get_data.load_pickle(f"lattice_cut/./T_C/bcc/", "all_gaps.pkl")

fig, ax = plt.subplots()

#for E_F in [-0.5, -0.4, -0.3, -0.2]:
#for omega_D in [0.02, 0.03, 0.04, 0.05]:
for U in [0.0, 0.01, 0.1, 0.2]:
    queried = main_df.query(f"N=={N} & U=={U} & E_F=={E_F} & omega_D=={omega_D} & g > 0.2").sort_values('g', ignore_index=True)
    gs = queried["g"]
    Deltas_enh = np.zeros_like(gs)

    for i, row in queried.iterrows():
        Deltas_enh[i] = abs(row["finite_gaps"][0][N//2])

    mask = Deltas_enh > 0.0
    ax.plot(gs[mask], np.log( Deltas_enh[mask] ), 
            label=rf"$\mu^* = {U}$"#rf"$\omega_\mathrm{{D}} = {2*omega_D}W$"#rf"$E_\mathrm{{F}} = {E_F}$" 
            )

ax.legend()
ax.set_xlabel(r"$g$")
ax.set_ylabel(r"$\ln (|\Delta (\varepsilon_\mathrm{Peak}) / W|)$")

plt.show()