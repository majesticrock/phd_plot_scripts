import matplotlib.pyplot as plt
import numpy as np
import __path_appender as __ap
from matplotlib.gridspec import GridSpec
__ap.append()
from get_data import *
import os
from create_figure import *
import string

systems = ["sc", "bcc", "fcc"]

Gs = [1.0, 1.2, 1.4, 1.6]
Ef = 0.

fig = create_figure(0.7, 0.8, generator=plt.figure)
gs = GridSpec(2, 3, hspace=0.4, wspace=0.15, height_ratios=[1.2, 1])
axes_energy = [ fig.add_subplot(gs[0, i]) for i in range(3) ]
ax_gap = fig.add_subplot(gs[1, :])

axes_energy[1].set_xlabel(r"$\varepsilon \times 10^2$", labelpad=0.5)
axes_energy[0].set_ylabel(r"$(E(\varepsilon) - \Delta_\mathrm{max}) \times 10^2$")
axes_energy[1].set_yticklabels([])
axes_energy[2].set_yticklabels([])

import colorsys
import matplotlib.colors as mcolors
def adjust_brightness(color, target):
    r, g, b = color
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # Convert back to RGB for each new lightness
    return colorsys.hls_to_rgb(h, target, s)

def rgb_to_grayscale(color):
    return (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2],) * 3

for i, ax, system in zip([0, 1, 2], axes_energy, systems):
    base_color_rgb = mcolors.to_rgb(plt.cm.tab10(i))
    colors_close_to_C = [
        adjust_brightness(base_color_rgb, target) for target in np.linspace(0.2, 0.75, 4)
    ]
    
    for color, g in zip(colors_close_to_C, Gs):
        main_df = load_panda("lattice_cut", f"./{system}", "gap.json.gz",
                            **lattice_cut_params(N=16000, 
                                                 g=g, 
                                                 U=0, 
                                                 E_F=Ef,
                                                 omega_D=0.02))
        energy_space = main_df["energies"]
        mask = (energy_space < 0.08) & (energy_space > -0.08)
        ax.plot(1e2 * energy_space[mask], 1e2 * (np.sqrt((energy_space - main_df["E_F"])**2 + main_df["Delta"]**2)[mask] - np.max(main_df["Delta"])), 
                label=f"${g}$", color=color)

    ax.annotate(
            f"({string.ascii_lowercase[i]})",
            xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), textcoords='offset fontsize', 
            verticalalignment='top', fontfamily='serif', bbox=dict(facecolor='1.0', edgecolor='black', pad=3))

    ax.set_ylim(-0.8, 1)

grayscale_colors = [rgb_to_grayscale(c) for c in colors_close_to_C]
legend_labels = [f"${g}$" for g in Gs]
legend_handles = [
    plt.Line2D([0], [0], color=grayscale_colors[i], label=legend_labels[i])
    for i in range(len(grayscale_colors))
]
leg = axes_energy[2].legend(handles=legend_handles, loc="lower left", title="$g$", 
                      ncols=1, 
                      bbox_to_anchor=(1.134, 0.0, 0.8, 1), 
                      columnspacing=1,
                      handlelength=1,
                      fancybox=False,
                      edgecolor="inherit",
                      framealpha=1,
                      #alignment="right",
                      borderaxespad=0,
                      handletextpad=0.4
                      )

ax_pos_diff = axes_energy[1].get_position().x0 - axes_energy[0].get_position().x0
for i, ax in enumerate(axes_energy): # Shrink axes boxes so that the legend fits besides them
    box = ax.get_position()
    ax.set_position([axes_energy[0].get_position().x0 + 0.8 * i * ax_pos_diff, box.y0, box.width * 0.8, box.height])

fontsize = fig.canvas.get_renderer().points_to_pixels(leg._fontsize)
pad = 2 * (leg.borderaxespad + leg.borderpad) * fontsize
leg._legend_box.set_height(leg.get_bbox_to_anchor().height-pad)

width_upper_row = axes_energy[0].get_position().x0 + 3.1 * 0.8 * ax_pos_diff
adjusted_leg_width = ax_gap.get_position().x0 + ax_gap.get_position().width - width_upper_row
leg._legend_box.set_width(adjusted_leg_width * fig.get_size_inches()[0] * fig.dpi)

##### Delta_true plot
labels = [
    r"(a)",
    r"(b)",
    r"(c)"
]
datas = [
    load_all("lattice_cut/sc/N=16000",  "resolvents.json.gz"),
    load_all("lattice_cut/bcc/N=16000", "resolvents.json.gz"),
    load_all("lattice_cut/fcc/N=16000", "resolvents.json.gz")
]

for i, (data, label) in enumerate(zip(datas, labels)):
    query = data.query(f"omega_D == 0.02 & g >= 0.2 & g <= 2.5 & Delta_max > 0", engine="python").sort_values("g")
    y_data = np.array([0.5 * boundaries[0] for boundaries in query["continuum_boundaries"]])
    ax_gap.plot(query["g"], (1 - y_data / query["Delta_max"]), label=label, color=f"C{i}")

ax_gap.legend()
ax_gap.set_xlabel("$g$")
ax_gap.set_ylabel(r"$1 - \Delta_\mathrm{true} / \Delta_\mathrm{max}$")
ax_gap.text(2.25, 0.2, "(d)")

fig.align_ylabels([axes_energy[0], ax_gap])

fig.savefig("build/maxtrue.pdf")
plt.show()