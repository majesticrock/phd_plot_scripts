from mrock.get_data import *
data_loader = DataLoader()
from mrock_centralized_scripts.legend import  *
import matplotlib.pyplot as plt
import mrock.continued_fraction as cf
import matplotlib as mpl

N=16000
OMEGA_D=0.02
E_F=-0.5
DOS="bcc"
G=1.5
n_mode = 0

main_df = data_loader.load_pickle(f"lattice_cut", f"{DOS}/N={N}", "resolvents.pkl").query(
    f"E_F == {E_F} & omega_D == {OMEGA_D} & g == {G} & U>0.1").sort_values("U", ignore_index=True)
Us = np.array([0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0])
main_df = main_df[main_df["U"].isin(Us)]

norm = mpl.colors.PowerNorm(
    0.5,
    vmin=Us.min(),
    vmax=Us.max()
)
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "inferno_truncated",
    plt.get_cmap("inferno")(np.linspace(0, 0.9, 256))
)

w_lin = np.linspace(0, 0.45, 2000, dtype=complex)
w_lin += 1e-4j

fig, ax = plt.subplots(layout="constrained")
ax.set_xlabel(r"$\omega / W$")
ax.set_ylabel(r"$U^2 \times \mathcal{A}_\mathrm{Higgs} (\omega) / W^{-1}$")

collected_higgs = []

for i, row in main_df.iterrows():
    if row["U"] < 0.3:
        resolvents = cf.ContinuedFraction(row, ignore_first=590, ignore_last=600, messages=False)
    else:
        resolvents = cf.ContinuedFraction(row, ignore_first=320, ignore_last=350, messages=False)
    A_higgs = resolvents.spectral_density(w_lin, "phase_SC", with_terminator=True)
    collected_higgs.append(A_higgs.tolist())
    ax.plot(w_lin.real, row["U"]**2 * A_higgs, c=cmap(norm(row["U"])))

data_dict = {"omegas": w_lin.real,
             "Us" : Us,
             "A_Higgs" : collected_higgs}

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # compatibility with older matplotlib versions
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("$U$")

ax.set_ylim(0, 0.025)

plt.show()