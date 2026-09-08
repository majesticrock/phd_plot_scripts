import matplotlib.pyplot as plt
import numpy as np
from mrock.get_data import DataLoader, lattice_cut_params

data_loader = DataLoader()

SYSTEM = "sc"

params = lattice_cut_params(
    N=16000,
    g=1.0,
    U=0.0,
    E_F=0.0,
    omega_D=0.02,
)

datasets = {
    "My Interaction": f"./{SYSTEM}",
    "LW Interaction": f"LW/{SYSTEM}",
}

fig, ax = plt.subplots(figsize=(7, 5))

for label, folder in datasets.items():
    main_df = data_loader.load_panda(
        "lattice_cut",
        folder,
        "gap.json.gz",
        **params,
    )

    energy_space = main_df["energies"]
    delta = main_df["Delta"]

    ax.plot(energy_space, delta, label=label)

    print(
        label,
        f"Max = {np.max(delta):.5f}, "
        f"Min = {np.min(delta):.5f}",
    )

ax.set_xlabel(r"$\epsilon - \mu$")
ax.set_ylabel(r"$\Delta$")
ax.legend()

fig.tight_layout()
plt.show()