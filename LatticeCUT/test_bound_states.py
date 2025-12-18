import numpy as np
import __path_appender as __ap
__ap.append()

from get_data import *

import cpp_continued_fraction as ccf

SYSTEM = 'sc'
main_df = load_panda("lattice_cut", f"./{SYSTEM}", "resolvents.json.gz",
                    **lattice_cut_params(N=16000, 
                                         g=2.5, 
                                         U=0.0, 
                                         E_F=0,
                                         omega_D=0.02))
a_inf = (main_df["continuum_boundaries"][0]**2 + main_df["continuum_boundaries"][1]**2) * 0.5
b_inf = (main_df["continuum_boundaries"][1]**2 - main_df["continuum_boundaries"][0]**2) * 0.25
A = main_df["resolvents.phase_SC"][0]["a_i"]
B = main_df["resolvents.phase_SC"][0]["b_i"]

terminate = True

def g_tail(z, a_inf, b_inf):
    if terminate:
        return (z - a_inf - np.sign(z.real - a_inf) * np.sqrt((z - a_inf)**2 - 4*b_inf**2)) / (2*b_inf**2)
    return 0.0

def cf_denominator(z, A, B, k0, a_inf, b_inf):
    D = z - A[k0] - B[k0 + 1] * g_tail(z, a_inf, b_inf)
    for k in range(k0 - 1, -1, -1):
        D = z - A[k] - B[k+1] / D
    return D / B[0]

from scipy.optimize import brentq

omega_minus = main_df["continuum_boundaries"][0]
z_max = omega_minus**2

def find_bound_states(A, B, a_inf, b_inf, k0,
                      z_min=0.0, z_max=z_max,
                      n_scan=2000):
    zs = np.linspace(z_min, z_max, n_scan)
    vals = np.array([cf_denominator(z, A, B, k0, a_inf, b_inf) for z in zs])

    roots = []
    for i in range(len(zs) - 1):
        if np.sign(vals[i]) != np.sign(vals[i+1]):
            z1, z2 = zs[i], zs[i+1]
            try:
                zb = brentq(
                    cf_denominator,
                    z1, z2,
                    args=(A, B, k0, a_inf, b_inf),
                    maxiter=200
                )
                roots.append(zb)
            except ValueError:
                print("ERROR")
                pass
    return np.unique(np.array(roots))

def bound_state_weight(zb, A, B, k0, a_inf, b_inf, dz=1e-8):
    Dp = cf_denominator((zb + dz)**2, A, B, k0, a_inf, b_inf)
    Dm = cf_denominator((zb - dz)**2, A, B, k0, a_inf, b_inf)
    dDdz = (Dp - Dm) / (2*dz)
    return 1. / dDdz


k0 = 150  # conservative, stable
z_bound = find_bound_states(A, B, a_inf, b_inf, k0)

print("Bound states (omega, weight):")
for zb in z_bound:
    omega_b = np.sqrt(zb)
    weight = bound_state_weight(omega_b, A, B, k0, a_inf, b_inf)
    if weight < 0:
        continue
    print(f"ω = {omega_b:.12e},   Z = {weight:.6e}")

state_info_list = []
print("")

for k_term in (k0 + np.arange(-10, 10, 1)):
    cf_data = ccf.ContinuedFractionData(a_inf, b_inf**2, 
                                        np.array([main_df["continuum_boundaries"][0]**2, main_df["continuum_boundaries"][1]**2]), 
                                        A, B, k0, terminate)
    state_info = ccf.classify_bound_states(cf_data, 2000, 1e-8, 48, 200)
    state_info_list.append(state_info)

#for info in state_info:
#    if info[1] < 1e-8 and info[0] > 1e-8:
#        continue
#    print(f"ω = {info[0]:.12e},   Z = {info[1]:6e}")
#print(ccf.denominator(0.0, cf_data), cf_denominator(0.0j, A, B, k0, a_inf, b_inf))

