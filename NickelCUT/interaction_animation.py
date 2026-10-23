from pathlib import Path

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
    "cmap" :          "seismic"
}

p = Momentum(L, 0, L//2)

L_MIN = 0.4
L_MAX = 0.6

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

        max_l = np.max(l_values)
        if L_MIN > max_l:
            raise ValueError(f"L_MIN={L_MIN} exceeds the maximum available l-value ({max_l}).")

        upper_l = min(L_MAX, max_l)
        self.frame_indices = np.flatnonzero((l_values >= L_MIN) & (l_values <= upper_l))
        if self.frame_indices.size == 0:
            raise ValueError(f"No available l-values fall in the range [{L_MIN}, {upper_l}].")

        self.matrices = [[], [], []]
        for flow_data in data["extracted_channels"]:
            self.matrices[0].append(
                flow_data["superconductivity"][p.pos].reshape(L, L) * N
            )
            self.matrices[1].append(
                (flow_data["density_wave_differing"][p.pos]
                 - flow_data["density_wave_same"][p.pos]).reshape(L, L) * N
            )
            self.matrices[2].append(
                (flow_data["single_particle_energy_differing"][p.pos]
                 + flow_data["single_particle_energy_same"][p.pos]).reshape(L, L) * N
            )

        vmax = max(
            np.max(np.abs(channel_matrices[index]))
            for channel_matrices in self.matrices
            for index in self.frame_indices
        )
        if vmax == 0.0:
            vmax += 0.1

        fig, axes = plt.subplots(1, len(CHANNELS), figsize=(15, 5), layout="constrained")
        self.images = [
            ax.imshow(
                self.matrices[channel_index][self.frame_indices[0]],
                vmin=-vmax,
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

        self.title = fig.suptitle(f"ELL_STEP={0}, $\\ell={data['l_times'][0]:.3g}$")
        fig.colorbar(self.images[0], ax=axes, label=r"$V(k_x,k_y)$")

        self.animation = FuncAnimation(
            fig,
            self.update,
            frames=self.frame_indices,
            interval=600,
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

#pa.animation.save(
#    str(Path(__file__).with_name("interaction_animation.mp4")),
#    writer="ffmpeg",
#    fps=2,
#)

plt.show()