import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from create_momentum_labels import create_momentum_labels
from load_flow_files import load_flow_file
from nickel_cut_parameters import FLOW_PARAMETERS

from Momentum import Momentum

data = load_flow_file(**FLOW_PARAMETERS, resume_num="", force_json=False, dense=True)

L = data["L"]
N = L * L

im_show_kwargs = {
    "origin":         "lower",
    "aspect":         "equal",
    "interpolation" : "nearest",
    "cmap" :          "viridis"
}

p = Momentum(L, 0, L//2)

L_MIN = 0
L_MAX = 13

CHANNELS = (
    "SC",
    "AFM",
    "HF",
)

class PauseAnimation:
    def __init__(self):
        l_values = np.asarray(data["l_times"])
        if len(l_values) != len(data["extracted_channels"]):
            raise ValueError("l_times and extracted_channels must have equal lengths.")

        max_l = l_values[-2]
        if L_MIN > max_l:
            raise ValueError(f"L_MIN={L_MIN} exceeds the maximum available l-value ({max_l}).")

        upper_l = min(L_MAX, max_l)
        self.frame_indices = np.flatnonzero((l_values >= L_MIN) & (l_values <= upper_l))
        if self.frame_indices.size == 0:
            raise ValueError(f"No available l-values fall in the range [{L_MIN}, {upper_l}].")

        self.matrices = [[], [], []]
        for i in range(len(data["extracted_channels"])-1):
            early = data["extracted_channels"][i]
            late = data["extracted_channels"][i+1]
            dl = data["l_times"][i+1] - data["l_times"][i]
            
            self.matrices[0].append(
                np.abs(late["superconductivity"][p.pos]
                    - early["superconductivity"][p.pos]
                ).reshape(L, L) * N / dl
            )
            self.matrices[1].append(
                np.abs(late["density_wave_differing"][p.pos] - late["density_wave_same"][p.pos]
                    - early["density_wave_differing"][p.pos] - early["density_wave_same"][p.pos]
                ).reshape(L, L) * N / dl
            )
            self.matrices[2].append(
                np.abs(late["single_particle_energy_differing"][p.pos] #+ late["single_particle_energy_same"][p.pos]
                    - early["single_particle_energy_differing"][p.pos] #+ early["single_particle_energy_same"][p.pos]
                ).reshape(L, L) * N / dl
            )

        self.l_inf_norms = [
            np.asarray([np.max(np.abs(matrix)) for matrix in channel_matrices])
            for channel_matrices in self.matrices
        ]
        self.l2_rms_norms = [
            np.asarray([np.sqrt(np.mean(matrix**2)) for matrix in channel_matrices])
            for channel_matrices in self.matrices
        ]

        vmax = max(
            np.max(np.abs(channel_matrices[index]))
            for channel_matrices in self.matrices
            for index in self.frame_indices
        )
        if vmax == 0.0:
            vmax += 0.1
        vmax = 4
        vmin = min(
            np.min(np.abs(channel_matrices[index]))
            for channel_matrices in self.matrices
            for index in self.frame_indices
        )

        fig = plt.figure(figsize=(15, 9), layout="constrained")
        grid = fig.add_gridspec(2, len(CHANNELS), height_ratios=(2, 1))
        axes = [fig.add_subplot(grid[0, index]) for index in range(len(CHANNELS))]
        norm_ax = fig.add_subplot(grid[1, :])
        self.images = [
            ax.imshow(
                self.matrices[channel_index][self.frame_indices[0]],
                vmin=vmin,
                vmax=vmax,
                **im_show_kwargs,
            )
            for channel_index, ax in enumerate(axes)
        ]

        ticks, labels = create_momentum_labels(L)
        for ax, channel in zip(axes, CHANNELS):
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
            ax.set_xlabel(r"$k_x$")
            ax.set_title(channel)
        axes[0].set_ylabel(r"$k_y$")

        colors = plt.get_cmap("tab10").colors
        for channel_index, (channel, color) in enumerate(zip(CHANNELS, colors)):
            norm_ax.plot(
                l_values[:-1],
                self.l_inf_norms[channel_index],
                color=color,
                label=rf"{channel} $L_\infty$",
            )
            norm_ax.plot(
                l_values[:-1],
                self.l2_rms_norms[channel_index],
                color=color,
                linestyle="--",
                label=rf"{channel} $L_2/\sqrt{{N}}$",
            )
        norm_ax.set_xlabel(r"$\ell$")
        norm_ax.set_ylabel(r"Norm of $|\partial_\ell V|$")
        norm_ax.grid(True, alpha=0.3)
        norm_ax.set_ylim(0, None)
        norm_ax.set_xlim(l_values[0], l_values[-2])
        norm_ax.legend(ncol=3)

        self.title = fig.suptitle("")
        fig.colorbar(self.images[0], ax=axes, label=r"$|\partial \ell V(k_x,k_y)|$")

        self.animation = FuncAnimation(
            fig,
            self.update,
            frames=self.frame_indices,
            interval=400,
            blit=False,
            repeat=True,
        )
        
        self.paused = False

        fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused

    def update(self, ell_step):
        for image, channel_matrices in zip(self.images, self.matrices):
            image.set_data(channel_matrices[ell_step])
        self.title.set_text(f"ELL_STEP={ell_step}, $\\ell={data['l_times'][ell_step]:.3g}$")
        return (*self.images, self.title)

pa = PauseAnimation()

plt.show()