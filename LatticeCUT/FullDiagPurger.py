import numpy as np

def __fill_temps__(eigenvalues, weights, first_eigenvectors):
    temp_evs = []
    temp_weights = []
    temp_vectors = []
    for ev, weight, vector in zip(eigenvalues, weights, first_eigenvectors):
        if len(temp_evs) == 0:
            temp_evs.append(ev)
            temp_weights.append(weight)
            temp_vectors.append(vector)
        elif abs(ev - temp_evs[-1]) > 1e-5:
            temp_evs.append(ev)
            temp_weights.append(weight)
            temp_vectors.append(vector)
        elif weight > temp_weights[-1]:
            temp_evs[-1] = ev
            temp_weights[-1] = weight
            temp_vectors[-1] = vector
    return temp_evs, temp_weights, temp_vectors

class FullDiagPurger:
    def __init__(self, system_data):
        temp_amplitude_evs, temp_amplitude_weights, temp_amplitude_vectors = __fill_temps__(
            system_data["amplitude.eigenvalues"], 
            system_data["amplitude.weights"][0], 
            system_data["amplitude.first_eigenvectors"]
        )

        temp_phase_evs, temp_phase_weights, temp_phase_vectors = __fill_temps__(
            system_data["phase.eigenvalues"], 
            system_data["phase.weights"][0], 
            system_data["phase.first_eigenvectors"]
        )
        
        doublets = []
        for i, a_ev in enumerate(temp_amplitude_evs):
            for j, p_ev in enumerate(temp_phase_evs):
                if abs(a_ev - p_ev) < 1e-5:
                    doublets.append((i, j))

        correct_amplitude_indices = [ i for i in range(len(temp_amplitude_evs)) if all(i != dt[0] for dt in doublets) ]
        correct_phase_indices = [ j for j in range(len(temp_phase_evs)) if all(j != dt[1] for dt in doublets) ]

        for doublet in doublets:
            a_i, p_i = doublet
            if temp_amplitude_weights[a_i] >= temp_phase_weights[p_i]:
                correct_phase_indices.append(p_i)
            else:
                correct_amplitude_indices.append(a_i)
        
        correct_amplitude_indices.sort()
        correct_phase_indices.sort()
        
        self.amplitude_eigenvalues = np.array([ temp_amplitude_evs[i] for i in correct_amplitude_indices ])
        self.amplitude_weights =     np.array([ temp_amplitude_weights[i] for i in correct_amplitude_indices ])
        self.amplitude_first_eigenvectors = np.array([ temp_amplitude_vectors[i] for i in correct_amplitude_indices ])
        
        self.phase_eigenvalues = np.array([ temp_phase_evs[i] for i in correct_phase_indices ])
        self.phase_weights =     np.array([ temp_phase_weights[i] for i in correct_phase_indices ])
        self.phase_first_eigenvectors = np.array([ temp_phase_vectors[i] for i in correct_phase_indices ])