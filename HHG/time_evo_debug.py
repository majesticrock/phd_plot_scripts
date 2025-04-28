import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

import __path_appender
__path_appender.append()
from get_data import *
from legend import *

# Load the main dataframe
main_df = load_panda("HHG", "test_200/cosine_laser/PiFlux", "time_evolution.json.gz", 
                     **hhg_params(T=0, E_F=0, v_F=1.5e5, band_width=400, field_amplitude=1.6, photon_energy=5.25, decay_time=-1))

frequencies = main_df["frequencies"]
times = np.linspace(0, 8, len(main_df["time_evolutions"][0]))

# Set up colormap for line colors
num_lines = len(main_df["time_evolutions"])
cmap = get_cmap("gist_rainbow")  # Use the 'viridis' colormap (you can choose another one)
norm = Normalize(vmin=0, vmax=num_lines - 1)  # Normalize dataset indices to [0, 1]
colors = cmap(norm(range(num_lines)))  # Generate colors for each dataset index

fig, axes = plt.subplots(nrows=2, figsize=(6.4, 9.6))

for i in range(num_lines):
    # Plot time evolution data with colored lines
    axes[0].plot(times, main_df["time_evolutions"][i], color=colors[i], rasterized=True)
    
    # Compute current density and plot with colored lines
    current_density = frequencies[::8] * (main_df["debug_fft_real"][i] + 1.0j * main_df["debug_fft_imag"][i])
    current_density += 1.0j * main_df["time_evolutions"][i][-1] * np.exp(1.0j * 16. * np.pi * frequencies[::8])
    
    axes[1].plot(frequencies[::8], np.abs(current_density), color=colors[i], rasterized=True)
    axes[1].set_yscale("log")

# Add labels to both plots
axes[0].set_xlabel(legend(r"t \omega_L / (2\pi)"))
axes[0].set_ylabel(legend(r"\sigma^z (t)"))

axes[1].set_xlabel(legend(r"\omega / \omega_L"))
axes[1].set_ylabel(legend(r"\sigma^z (\omega)", "normalized"))

# Create a colorbar for the line colors
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # ScalarMappable for color mapping
sm.set_array([])  # Dummy array for compatibility with colorbar creation

fig.colorbar(sm, ax=axes.ravel().tolist(), label="Dataset Index")  # Add shared colorbar to both subplots

plt.show()