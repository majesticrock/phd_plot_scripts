import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from get_data import *
from legend import *
import matplotlib.pyplot as plt
import continued_fraction_pandas as cf
import matplotlib as mpl

N=16000
OMEGA_D=0.02
E_F=-0.5
DOS="bcc"
G=1.5
n_mode = 0

main_df = load_pickle(f"lattice_cut/{DOS}/N={N}", "resolvents.pkl").query(
    f"E_F == {E_F} & omega_D == {OMEGA_D} & g == {G} & U>=1").sort_values("U", ignore_index=True)

color_values = main_df["U"].to_numpy()
norm = mpl.colors.Normalize(
    vmin=color_values.min(),
    vmax=color_values.max()
)
cmap = plt.get_cmap("inferno")
n_lines = len(main_df.index)

w_lin = np.linspace(0, 0.5, 2000, dtype=complex)
w_lin += 1e-4j

fig, ax = plt.subplots()
ax.set_xlabel(r"$\omega / W$")
ax.set_ylabel(r"$\mathcal{A} (\omega) / W^{-1}$")

for i, row in main_df.iterrows():
    resolvents = cf.ContinuedFraction(row, ignore_first=260, ignore_last=280)
    A_higgs = resolvents.spectral_density(w_lin, "amplitude_SC", withTerminator=True)
    ax.plot(w_lin.real, row["U"]**2 * A_higgs, c=cmap(i / n_lines))
    
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # compatibility with older matplotlib versions
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("$U$")

ax.set_ylim(0, 0.025)

plt.show()