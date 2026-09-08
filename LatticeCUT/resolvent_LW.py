import numpy as np
import matplotlib.pyplot as plt
from mrock.get_data import DataLoader, lattice_cut_params
import mrock.continued_fraction as cf

data_loader = DataLoader()

SYSTEM = "sc"

params = lattice_cut_params(
    N=16000,
    g=2,
    U=0.0,
    E_F=0,
    omega_D=0.02,
)

# Replace these folder names with the two datasets to compare.
datasets = {
    "My Interaction": f"./{SYSTEM}",
    "LW Interaction": f"LW/{SYSTEM}",
}

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 8))

axes[0].set_ylabel(r"$\mathcal{A}_{\mathrm{phase}}(\omega) / W^{-1}$")
axes[1].set_ylabel(r"$\mathcal{A}_{\mathrm{amplitude}}(\omega) / W^{-1}$")
axes[1].set_xlabel(r"$\omega / W$")

for label, folder in datasets.items():
    main_df = data_loader.load_panda(
        "lattice_cut",
        folder,
        "resolvents.json.gz",
        **params,
    )

    resolvents = cf.ContinuedFraction(
        main_df,
        ignore_first=80,
        ignore_last=120,
    )

    print(label, "Delta_true =", resolvents.continuum_edges()[0])

    w_lin = np.linspace(
        0,
        0.5 * main_df["continuum_boundaries"][1],
        10000,
        dtype=complex,
    )
    w_lin += 1e-4j

    A_phase = resolvents.spectral_density(
        w_lin,
        "phase_SC",
        with_terminator=True,
    )
    A_amplitude = resolvents.spectral_density(
        w_lin,
        "amplitude_SC",
        with_terminator=True,
    )

    axes[0].plot(w_lin.real, A_phase, label=label)
    axes[1].plot(w_lin.real, A_amplitude, label=label)

for ax in axes:
    ax.set_ylim(-0.05, 3.5)
    ax.legend()

fig.tight_layout()
plt.show()