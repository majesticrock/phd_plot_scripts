import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

# Load the main dataframe
main_df = load_panda("HHG", "test/cosine_laser/PiFlux", "time_evolution.json.gz", 
                     **hhg_params(T=300, E_F=118, v_F=1.5e6, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))

frequencies = main_df["frequencies"][::8]
times = np.linspace(0, 8, len(main_df["time_evolutions"][0]))

# Prepare data for heatmaps
time_evolution_data = np.array(main_df["time_evolutions"])  # Shape: (num_datasets, num_time_points)
current_density_data = []

for i in range(len(main_df["time_evolutions"])):
    current_density = frequencies * (main_df["debug_fft_real"][i] + 1.0j * main_df["debug_fft_imag"][i])
    current_density += 1.0j * main_df["time_evolutions"][i][-1] * np.exp(1.0j * 16. * np.pi * frequencies)
    current_density_data.append(np.abs(current_density))  # Store magnitude of complex values

current_density_data = np.array(current_density_data)  # Shape: (num_datasets, num_frequencies)

# Create subplots for heatmaps
fig, axes = plt.subplots(nrows=2, figsize=(6.4, 9.6))

# Heatmap for time evolution data
im_time = axes[0].imshow(time_evolution_data.T, aspect='auto', extent=[0, len(main_df["time_evolutions"]), times[0], times[-1]], cmap='viridis')
axes[0].set_xlabel(legend(r"Dataset Index"))
axes[0].set_ylabel(legend(r"t \omega_L / (2\pi)"))
fig.colorbar(im_time, ax=axes[0], label="Time Evolution Magnitude")

# Heatmap for current density data
im_freq = axes[1].imshow(current_density_data.T, aspect='auto', extent=[0, len(main_df["time_evolutions"]), frequencies[0], frequencies[-1]], cmap='viridis', norm=mc.LogNorm())
axes[1].set_xlabel(legend(r"Dataset Index"))
axes[1].set_ylabel(legend(r"\omega / \omega_L"))
fig.colorbar(im_freq, ax=axes[1], label="Current Density Magnitude")

fig.tight_layout()
plt.show()