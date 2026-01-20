import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
from get_data import *

SYSTEM = 'fcc'
N=4000
params = lattice_cut_params(N=N, 
                            g=2.5,
                            U=0.0, 
                            E_F=-0.5,
                            omega_D=0.02)
main_df = load_panda("lattice_cut", f"test/{SYSTEM}", "full_diagonalization.json.gz", **params)

def compute_resolvent(evs, weights, z):
    resolvent = np.zeros_like(z, dtype=np.complex128)
    for ev, weight in zip(evs, weights):
        resolvent += weight / (z**2 - ev**2)
    return resolvent

fig_r, axes_r = plt.subplots(nrows=2, sharex=True, sharey=True)
fig_r.subplots_adjust(hspace=0)
axes_r[0].set_ylabel("Higgs")
axes_r[1].set_ylabel("Phase")
axes_r[1].set_xlabel(r"$\omega$")

evs = np.asarray(main_df["amplitude.eigenvalues"])
z = np.linspace(0, np.max(evs) * 1.01, 20000) + 4e-4j
axes_r[0].plot(z.real, -np.imag(compute_resolvent(evs, np.asarray(main_df["amplitude.weights"][0]), z)))


evs = np.asarray(main_df["phase.eigenvalues"])
axes_r[1].plot(z.real, -np.imag(compute_resolvent(evs, np.asarray(main_df["phase.weights"][0]), z)))

for i, (e, w) in enumerate(zip(np.asarray(main_df["amplitude.eigenvalues"]), main_df["amplitude.weights"][0][:12])):
    print(i, f"- ev: {e}, weight: {w}")
print("########################")
for i, (e, w) in enumerate(zip(np.asarray(main_df["phase.eigenvalues"]), main_df["phase.weights"][0][:12])):
    print(i, f"- ev: {e}, weight: {w}")

for ax in axes_r:
    ax.axvline(main_df["continuum_boundaries"][0], ls="--", c="k")
    ax.axvline(main_df["continuum_boundaries"][1], ls="--", c="k")

axes_r[0].set_ylim(0, 15.5)

fig_wv, axes_wv = plt.subplots(nrows=3, sharex=True, sharey=True)
fig_wv.subplots_adjust(hspace=0)
axes_wv[0].set_ylabel("Higgs")
axes_wv[1].set_ylabel("Occupation")
axes_wv[2].set_ylabel("Phase")
axes_wv[-1].set_xlabel(r"$\varepsilon$")
epsilon = np.linspace(-1, 1, N)

def add_line(ax, y, **kwargs):
    y = np.asarray(y)
    if len(y) != N:
        return
    if abs(np.min(y)) > abs(np.max(y)):
        y = -y
    ax.plot(epsilon, y / np.max(np.abs(y)), **kwargs)

for i in range(len(main_df["amplitude.first_eigenvectors"])):
    if main_df["amplitude.eigenvalues"][i] > main_df["continuum_boundaries"][0]:
        continue
    if main_df["amplitude.weights"][0][i] < 1e-4:
        continue
    
    add_line(axes_wv[0], main_df["amplitude.first_eigenvectors"][i][:N], label=f"{i}")
    add_line(axes_wv[1], main_df["amplitude.first_eigenvectors"][i][N:])

for i in range(len(main_df["phase.first_eigenvectors"])):
    if main_df["phase.eigenvalues"][i] > main_df["continuum_boundaries"][0]:
        continue
    if main_df["phase.weights"][0][i] < 1e-4:
        continue
    add_line(axes_wv[2], main_df["phase.first_eigenvectors"][i], label=f"{i}")

axes_wv[0].legend(loc="upper right")
axes_wv[2].legend(loc="upper right")

plt.show()