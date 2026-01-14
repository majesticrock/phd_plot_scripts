import numpy as np
import __path_appender as __ap
__ap.append()

from get_data import *

import cpp_continued_fraction as ccf

SYSTEM = 'bcc'
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

k_min = 150
total_n_lines = 20
terminate = True
diffs = (A[k_min:-1] - a_inf)**2 / a_inf**2 + (np.sqrt(B[k_min + 1:]) - b_inf)**2 / b_inf**2
k0_candidates = np.argsort(diffs)[:total_n_lines] + k_min
print(k0_candidates)
state_info_list = []

for k0 in k0_candidates:
    cf_data = ccf.ContinuedFractionData(a_inf, b_inf**2, 
                                        np.array([main_df["continuum_boundaries"][0]**2, main_df["continuum_boundaries"][1]**2]), 
                                        A, B, k0, terminate)
    state_info = ccf.classify_bound_states(cf_data, 5000, 1e-8, 48, 200)
    state_info_list.append(state_info)

from math import sqrt

def associate_energies(data, tol):
    """
    data: list of lists of (energy, weight)
    tol:  energy matching tolerance

    returns: list of dicts with:
        energy, energy_error,
        weight, weight_error
    """

    points = [(e, w) for sub in data for (e, w) in sub]
    clusters = []

    for energy, weight in points:
        matched = False

        for c in clusters:
            if abs(energy - c["mean_energy"]) <= tol:
                c["energies"].append(energy)
                c["weights"].append(weight)
                matched = True
                break

        if not matched:
            clusters.append({
                "energies": [energy],
                "weights": [weight],
                "mean_energy": energy
            })

        # Update weighted energy mean
        for c in clusters:
            wsum = sum(c["weights"])
            c["mean_energy"] = sum(
                e * w for e, w in zip(c["energies"], c["weights"])
            ) / wsum

    results = []

    for c in clusters:
        energies = c["energies"]
        n = len(energies)
        weights = np.zeros(total_n_lines)
        weights[:n] = c["weights"]

        # --- Energy (position) ---
        wsum = sum(weights)
        mean_energy = sum(e * w for e, w in zip(energies, weights)) / wsum

        energy_variance = sum(
            w * (e - mean_energy) ** 2
            for e, w in zip(energies, weights)
        ) / wsum

        energy_error = sqrt(energy_variance) * (total_n_lines / n)

        # --- Weight ---
        mean_weight = sum(weights) / n

        weight_variance = sum(
            (w - mean_weight) ** 2 for w in weights
        ) / n

        weight_error = sqrt(weight_variance)

        results.append({
            "energy": mean_energy,
            "energy_error": energy_error,
            "weight": mean_weight,
            "weight_error": weight_error,
            "count": n
        })

    return results


out = associate_energies(state_info_list, tol=1e-3)
for r in out:
    #if r["count"] < 5:
    #    continue
    #total_uncertainty = max(r['weight_error'] / abs(r['weight']), r['energy_error'] / abs(r['energy']))
    #if total_certainty < 0.05:
    #    continue
    print(f"Energy = {r['energy']:.6e}, Uncertainty: {r['energy_error'] / abs(r['energy']):.4f}, "
          f"Weight = {r['weight']:.6e}, Uncertainty: {r['weight_error'] / abs(r['weight']):.4f}, "
          f"Count: {r['count']}")
