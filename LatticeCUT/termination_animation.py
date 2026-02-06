import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mrock_centralized_scripts.path_appender as __ap
__ap.append()
from create_zoom import *
from get_data import *
import continued_fraction_pandas as cf
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

SYSTEM = "bcc"
G   = 0.95
E_F = 0
OMEGA_D = 0.02
N = 16000

cmap = plt.get_cmap("jet")
step = 4
shift_range = np.arange(-10, 200, step)
init = 200

FWHM = 15. / step
sigma = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
W = int(FWHM)

fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8, 6))
fig.subplots_adjust(hspace=0, wspace=0)

main_df = load_panda(
    "lattice_cut",
    f"./{SYSTEM}",
    "resolvents.json.gz",
    **lattice_cut_params(
        N=N,
        g=G,
        U=0,
        E_F=E_F,
        omega_D=OMEGA_D,
    )
)

print(main_df["continuum_boundaries"][0])
w_lin = np.linspace(0.5 * main_df["continuum_boundaries"][0], 1.25 * main_df["continuum_boundaries"][0], 5000, dtype=complex)
w_lin += 1e-5j

resolvents = cf.ContinuedFraction(
    main_df,
    ignore_first=init,
    ignore_last=init + step
)

import time
begin = time.time()
# --- Precompute spectra ---
all_higgs = resolvents.spectral_density_varied_depth(
    w_lin, "amplitude_SC", shift_range, withTerminator=True
)
all_phase = resolvents.spectral_density_varied_depth(
    w_lin, "phase_SC", shift_range, withTerminator=True
)
end = time.time()
print("Duration:", end-begin)




# --- Local Gaussian-averaged mean ---
higgs_mean = gaussian_filter1d(all_higgs, sigma=sigma, axis=0, mode="nearest")
phase_mean = gaussian_filter1d(all_phase, sigma=sigma, axis=0, mode="nearest")
#higgs_mean = uniform_filter1d(all_higgs, size=W, axis=0, mode="nearest")
#phase_mean = uniform_filter1d(all_phase, size=W, axis=0, mode="nearest")


# --- Local Gaussian-averaged second moment ---
higgs_mean_sq = gaussian_filter1d(all_higgs**2, sigma=sigma, axis=0, mode="nearest")
phase_mean_sq = gaussian_filter1d(all_phase**2, sigma=sigma, axis=0, mode="nearest")
#higgs_mean_sq = uniform_filter1d(all_higgs**2, size=W, axis=0, mode="nearest")
#phase_mean_sq = uniform_filter1d(all_phase**2, size=W, axis=0, mode="nearest")

# --- Local variance ---
higgs_var = np.sqrt(np.clip(higgs_mean_sq - higgs_mean**2, 0.0, None))
phase_var = np.sqrt(np.clip(phase_mean_sq - phase_mean**2, 0.0, None))

# --- Local mean and variance artists ---
higgs_mean_line, = axes[0].plot([], [], c="k", lw=2, label="Local mean", alpha=0.5)
phase_mean_line, = axes[1].plot([], [], c="k", lw=2, alpha=0.5)

higgs_var_low_line, = axes[0].plot([], [], c="k", lw=1, ls="--", alpha=0.5)
higgs_var_upp_line, = axes[0].plot([], [], c="k", lw=1, ls="--", alpha=0.5)
phase_var_low_line, = axes[1].plot([], [], c="k", lw=1, ls="--", alpha=0.5)
phase_var_upp_line, = axes[1].plot([], [], c="k", lw=1, ls="--", alpha=0.5)





# --- Compute averages ---
avg_spectral_higgs = np.mean(all_higgs, axis=0)
avg_spectral_phase = np.mean(all_phase, axis=0)

# --- Find closest-to-average ---
deviations_higgs = np.zeros(len(all_higgs))
deviations_phase = np.zeros(len(all_higgs))
#for i, (h, p) in enumerate(zip(all_higgs, all_phase)):
#    np.linalg.norm(h - avg_spectral_higgs)
#    np.linalg.norm(p - avg_spectral_phase)

for i, (h, p, hm, pm) in enumerate(zip(higgs_var, phase_var, higgs_mean, phase_mean)):
    deviations_higgs[i] = np.linalg.norm(h / np.where(hm > 0, hm, 1)) + np.linalg.norm(h)
    deviations_phase[i] = np.linalg.norm(p / np.where(pm > 0, pm, 1)) + np.linalg.norm(p)

# --- Plot static reference curves ---
possible_higgs = deviations_higgs.argsort()
possible_phase = deviations_phase.argsort()
possible_higgs = possible_higgs[deviations_higgs[possible_higgs] <= 2 * np.min(deviations_higgs)]
possible_phase = possible_phase[deviations_phase[possible_phase] <= 2 * np.min(deviations_phase)]


best_higgs = np.min(possible_higgs)#np.argmin(deviations_higgs[:-int(FWHM)])
best_phase = np.min(possible_phase)#np.argmin(deviations_phase[:-int(FWHM)])
cta_h, = axes[0].plot( w_lin.real, all_higgs[best_higgs], c="k", ls=":", lw=2, label="Closest to average")
cta_p, = axes[1].plot( w_lin.real, all_phase[best_phase], c="k", ls=":", lw=2, label="Closest to average")

#print("Best termination:", init + shift_range[best_higgs], "and", init + shift_range[best_phase])

# --- Prepare animated line containers ---
higgs_lines = []
phase_lines = []

for _ in shift_range:
    lh, = axes[0].plot([], [])
    lp, = axes[1].plot([], [])
    higgs_lines.append(lh)
    phase_lines.append(lp)

# --- Axes formatting ---
for ax in axes:
    resolvents.mark_continuum(ax, label=None)
    ax.set_ylim(-0.05, 15)
    ax.set_xlim(w_lin.real.min(), w_lin.real.max())
    ax.set_xlabel(r"$\omega$")

axes[0].set_ylabel(r"$\mathcal{A}_\mathrm{Higgs}(\omega)$")
axes[1].set_ylabel(r"$\mathcal{A}_\mathrm{Phase}(\omega)$")
axes[0].legend(loc="upper right")

# --- Animation functions ---
def init():
    for lh, lp in zip(higgs_lines, phase_lines):
        lh.set_data([], [])
        lp.set_data([], [])
    return higgs_lines + phase_lines

def update(frame):
    color = cmap(frame / (len(shift_range) - 1))

    # --- Individual spectra ---
    higgs_lines[frame].set_data(w_lin.real, all_higgs[frame])
    higgs_lines[frame].set_color(color)

    phase_lines[frame].set_data(w_lin.real, all_phase[frame])
    phase_lines[frame].set_color(color)

    # --- Local mean ---
    higgs_mean_line.set_data(w_lin.real, higgs_mean[frame])
    phase_mean_line.set_data(w_lin.real, phase_mean[frame])

    higgs_var_low_line.set_data(w_lin.real, higgs_mean[frame] - higgs_var[frame])
    higgs_var_upp_line.set_data(w_lin.real, higgs_mean[frame] + higgs_var[frame])
    phase_var_low_line.set_data(w_lin.real, phase_mean[frame] - phase_var[frame])
    phase_var_upp_line.set_data(w_lin.real, phase_mean[frame] + phase_var[frame])

    # Static reference curves remain untouched
    return (higgs_lines[frame], phase_lines[frame], higgs_mean_line, phase_mean_line,
            higgs_var_low_line, higgs_var_upp_line, phase_var_low_line, phase_var_upp_line)



ani = FuncAnimation(
    fig,
    update,
    frames=len(shift_range),
    init_func=init,
    interval=150,
    blit=True,
    repeat=True,
)

plt.show()
