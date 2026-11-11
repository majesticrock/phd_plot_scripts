import matplotlib.pyplot as plt
import numpy as np

from finite_size_energy_gap import energy_gap_for_parameters
from load_flow_files import load_flow_file
from nickel_cut_parameters import FLOW_PARAMETERS


SYSTEM_SIZES = np.array([ 10, 12, 14, 16, 18 ])
CHANNELS = (
    ("extremal_sc_eigenvalues", "SC", "o"),
    ("extremal_charge_eigenvalues", "Charge", "s"),
    ("extremal_spin_eigenvalues", "Spin", "v"),
)

fig, ax = plt.subplots(layout="constrained")
extracted_data = np.zeros((3, len(SYSTEM_SIZES)))

for size_index, system_size in enumerate(SYSTEM_SIZES):
    parameters = {**FLOW_PARAMETERS, "L": system_size}
    data = load_flow_file(
        **parameters,
        resume_num="",
        force_json=False,
        dense=False,
    )
    print(f"Loaded L={system_size} data produced on {data['time']}")

    point_count = data["number_of_data_points"]
    l_times = np.asarray(data["l_times"][:point_count])
    eigenvalues = np.array(
        [data[key][:point_count] for key, _, _ in CHANNELS]
    )
    energy_gap = energy_gap_for_parameters(
        system_size,
        FLOW_PARAMETERS["tprime"],
        data["E_F"],
    )
    ell_threshold = 1.0 / energy_gap

    for channel_idx, channel_evs in enumerate(eigenvalues):
        extracted_data[channel_idx, size_index] = channel_evs[np.argmin(np.abs(l_times - ell_threshold))]

x_data = 1. / SYSTEM_SIZES
for data_row, (_, label, marker) in zip(extracted_data, CHANNELS):
    line, = ax.plot(x_data, data_row, marker=marker, label=label, markersize=10, ls="")
    
    mean = np.mean(data_row)
    std = np.std(data_row, mean=mean, ddof=1)
    ax.axhline(mean, c=line.get_color(), ls=":")
    ax.fill_between(x_data, mean - std, mean + std, color=line.get_color(), alpha=0.3)

ax.legend()
ax.set_xlabel(r"$1/L$")
ax.set_ylabel(r"$\lambda_\mathrm{ex}$")

plt.show()
