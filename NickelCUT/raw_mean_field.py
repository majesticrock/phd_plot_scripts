import numpy as np
import matplotlib.pyplot as plt

from Momentum import Momentum
from load_flow_files import load_flow_file, load_final_flow_state
from nickel_cut_parameters import FLOW_PARAMETERS

from copy import copy

L = FLOW_PARAMETERS["L"]
N = L * L
beta = 1. / (FLOW_PARAMETERS["T"] if FLOW_PARAMETERS["T"] > 0.0 else 0.01)
def fermi(energy):
    return 1.0 / (1.0 + np.exp(beta * energy))

data = load_flow_file(**FLOW_PARAMETERS, resume_num="", force_json=False, dense=True)
ELL_STEP = -1

dispersion      = data["extracted_channels"][ELL_STEP]["epsilon_tilde"]
self_energy     = data["extracted_channels"][ELL_STEP]["dispersion"] - dispersion
dw_channel      = data["extracted_channels"][ELL_STEP]["density_wave_differing"] - data["extracted_channels"][ELL_STEP]["density_wave_same"]
direct_channel  = data["extracted_channels"][ELL_STEP]["single_particle_energy_differing"] + data["extracted_channels"][ELL_STEP]["single_particle_energy_same"]

MOM_GAMMA = Momentum(L, L//2, L//2)
MOM_PI = Momentum(L, 0, 0)

#data = load_final_flow_state(**FLOW_PARAMETERS, resume_num="", force_json=True)
#dispersion      = data["epsilon_tilde"]
#self_energy     = data["dispersion"] - dispersion
#dw_channel      = 2 * (data["interactions_differing_spin"] - 2 * data["interactions_same_spin"])[:,:,MOM_PI.pos]
#direct_channel  = 2 * (data["interactions_differing_spin"] + 2 * data["interactions_same_spin"])[:,:,MOM_GAMMA.pos]


Delta       = np.random.rand(*self_energy.shape)
self_energy *= np.random.rand(*self_energy.shape)

mu = 0.
best_val = 1.
unique_energies = np.unique(dispersion + self_energy)
for i in range(len(unique_energies)-1):
    x = unique_energies[i]
    filling_error = np.abs(2. * np.average(fermi(dispersion + self_energy - x)) - data["filling"])
    if filling_error < best_val:
        best_val = copy(filling_error)
        mu       = copy(x)
    
    x = 0.5 * (unique_energies[i+1] + unique_energies[i])
    filling_error = np.abs(2. * np.average(fermi(dispersion + self_energy - x)) - data["filling"])
    if filling_error < best_val:
        best_val = copy(filling_error)
        mu       = copy(x)

print("mu =", mu, "    filling =", 2. * np.average(fermi(dispersion + self_energy - mu)),    "    trying to get", data["filling"])

def H_MF(k: Momentum):
    return np.array([
        [ dispersion[k.pos] + self_energy[k.pos] - mu,  Delta[k.pos] ],
        [ Delta[k.pos],                                 dispersion[(k+MOM_PI).pos] + self_energy[(k+MOM_PI).pos] - mu ]
    ])

AFM_expecs = np.zeros_like(Delta)
NUM_expecs = np.zeros_like(Delta)

def fill_expecs():
    for ix in range(L):
        for iy in range(L):
            k = Momentum(L, ix, iy)
            eigenvalues, eigenvectors = np.linalg.eigh(H_MF(k))
            rho = eigenvectors @ np.array([
                [1. - fermi(eigenvalues[0]), 0.],
                [0., 1. - fermi(eigenvalues[1])]
            ]) @ eigenvectors.T
            AFM_expecs[k.pos] = -rho[0,1]
            NUM_expecs[k.pos] = 1.-rho[0,0]

def compute_single_new(k: Momentum):
    _new_delta = 0.
    _new_self_energy = 0.
    for ix in range(L):
        for iy in range(L):
            _q = Momentum(L, ix, iy)
            _new_delta -= dw_channel[k.pos, _q.pos] * AFM_expecs[_q.pos]
            _new_self_energy += direct_channel[k.pos, _q.pos] * NUM_expecs[_q.pos]
    # factor 1/N is contained in the interaction itself
    return (_new_delta, _new_self_energy)


error = 100.
new_delta = np.zeros_like(Delta)
new_self_energy = np.zeros_like(Delta)
SIGMA = 0.5

while error > 1e-5:
    mu = 0.
    best_val = 1.
    unique_energies = np.unique(dispersion + self_energy)
    for i in range(len(unique_energies)-1):
        x = unique_energies[i]
        filling_error = np.abs(2. * np.average(fermi(dispersion + self_energy - x)) - data["filling"])
        if filling_error < best_val:
            best_val = copy(filling_error)
            mu       = copy(x)

        x = 0.5 * (unique_energies[i+1] + unique_energies[i])
        filling_error = np.abs(2. * np.average(fermi(dispersion + self_energy - x)) - data["filling"])
        if filling_error < best_val:
            best_val = copy(filling_error)
            mu       = copy(x)
    
    fill_expecs()
    
    for ix in range(L):
        for iy in range(L):
            k = Momentum(L, ix, iy)
            new_delta[k.pos], new_self_energy[k.pos] = compute_single_new(k)
    
    error = np.sqrt(np.sum((new_delta - Delta)**2 + (new_self_energy - self_energy)**2))
    Delta       = (1-SIGMA) * new_delta + SIGMA * Delta
    self_energy = (1-SIGMA) * new_self_energy + SIGMA * self_energy
    
    print("Delta_max =", Delta[np.argmax(np.abs(Delta))], "  Error =", error, "  Filling =", 2. * np.average(NUM_expecs))

fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
Delta = Delta.reshape(L, L)

vmin = np.min(Delta)
vmin = 0. if vmin > 0. else vmin
im0 = ax.imshow(Delta, origin="lower", cmap="magma", vmin=vmin, vmax=np.max(Delta))
ax.set_xlabel(r"$q_x$")
ax.set_ylabel(r"$q_y$")
fig.colorbar(im0, ax=ax, label=r"$\Delta(\mathbf{k})$")

ticks = np.array([0, L // 4, L // 2, 3 * L // 4])
ax.set_xticks(ticks)
ax.set_yticks(ticks)
tick_labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$"]
ax.set_xticklabels(tick_labels)



fig2, ax2 = plt.subplots(figsize=(7, 5), constrained_layout=True)
eps = (dispersion + self_energy - mu).reshape(L, L)

vmin2 = -max(abs(np.min(eps)), np.max(eps))
im2 = ax2.imshow(eps, origin="lower", cmap="seismic", vmin=vmin2, vmax=-vmin2)
ax2.set_xlabel(r"$q_x$")
ax2.set_ylabel(r"$q_y$")
fig2.colorbar(im2, ax=ax2, label=r"$\varepsilon(\mathbf{k})$")

ax2.set_xticks(ticks)
ax2.set_yticks(ticks)
ax2.set_xticklabels(tick_labels)


plt.show()