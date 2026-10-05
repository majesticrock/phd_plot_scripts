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

# density_wave_differing | density_wave_same
# single_particle_energy_differing | single_particle_energy_same
# superconductivity

class PauseAnimation:
    def __init__(self):
        fig, ax = plt.subplots()
        self.matrices = [
            #flow_data["superconductivity"][p.pos].reshape(L, L) * N
            (flow_data["density_wave_differing"][p.pos] - flow_data["density_wave_same"][p.pos]).reshape(L, L) * N
            #(flow_data["single_particle_energy_differing"][p.pos] + flow_data["single_particle_energy_same"][p.pos]).reshape(L, L) * N
            for flow_data in data["extracted_channels"]
        ]

        vmax = max(np.max(np.abs(V)) for V in self.matrices)
        if vmax == 0.0:
            vmax += 0.1
        self.im = ax.imshow(self.matrices[0], vmin=-vmax, vmax=vmax, **im_show_kwargs)

        # Set custom tick labels for momentum space
        ticks, labels = create_momentum_labels(L)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)

        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
        self.title = ax.set_title("0")
        fig.colorbar(self.im, label=r"$V(k_x,k_y)$")
        fig.tight_layout()

        self.animation = FuncAnimation(
            fig,
            self.update,
            frames=len(self.matrices),
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
        self.im.set_data(self.matrices[ell_step])
        self.title.set_text(f"ELL_STEP={ell_step}, $\\ell={data['l_times'][ell_step]:.3g}$")
        return self.im, self.title

pa = PauseAnimation()

plt.show()