import numpy as np

PEAK_DIFF_TOL = 0.005

def __fill_temps__(eigenvalues, weights, first_eigenvectors, continuum_edge):
    temp_evs = []
    temp_weights = []
    temp_vectors = []
    for ev, weight, vector in zip(eigenvalues, weights, first_eigenvectors):
        if ev >= continuum_edge or weight < 1e-7:
            continue
        if len(temp_evs) == 0:
            temp_evs.append(ev)
            temp_weights.append(weight)
            temp_vectors.append(vector)
        elif abs(ev - temp_evs[-1]) > PEAK_DIFF_TOL:
            temp_evs.append(ev)
            temp_weights.append(weight)
            temp_vectors.append(vector)
        elif weight > temp_weights[-1]:
            temp_evs[-1] = ev
            temp_weights[-1] = weight
            temp_vectors[-1] = vector
    return temp_evs, temp_weights, temp_vectors

def __plot_line__(ax, x, y, **plot_kwargs):
    ax.plot(x, y / np.max(np.abs(y)), **plot_kwargs)

class FullDiagPurger:
    def __init__(self, system_data, epsilon_space):
        temp_amplitude_evs, temp_amplitude_weights, temp_amplitude_vectors = __fill_temps__(
            system_data["amplitude.eigenvalues"], 
            system_data["amplitude.weights"][0], 
            system_data["amplitude.first_eigenvectors"],
            system_data["continuum_boundaries"][0]
        )

        temp_phase_evs, temp_phase_weights, temp_phase_vectors = __fill_temps__(
            system_data["phase.eigenvalues"], 
            system_data["phase.weights"][0], 
            system_data["phase.first_eigenvectors"],
            system_data["continuum_boundaries"][0]
        )
        self.epsilon_space = epsilon_space
        self.N = len(epsilon_space)
        
        doublets = []
        for i, a_ev in enumerate(temp_amplitude_evs):
            for j, p_ev in enumerate(temp_phase_evs):
                if abs(a_ev - p_ev) < PEAK_DIFF_TOL:
                    doublets.append((i, j))

        correct_amplitude_indices = [ i for i in range(len(temp_amplitude_evs)) if all(i != dt[0] for dt in doublets) ]
        correct_phase_indices = [ j for j in range(len(temp_phase_evs)) if all(j != dt[1] for dt in doublets) ]
        glimmering_amplitude_indices = []
        glimmering_phase_indices = []
        
        for doublet in doublets:
            a_i, p_i = doublet
            if temp_amplitude_weights[a_i] < temp_phase_weights[p_i]:
                correct_phase_indices.append(p_i)
                glimmering_amplitude_indices.append(a_i)
            else:
                correct_amplitude_indices.append(a_i)
                glimmering_phase_indices.append(p_i)
        
        correct_amplitude_indices.sort()
        correct_phase_indices.sort()
        
        self.amplitude_eigenvalues        = np.array([ temp_amplitude_evs[i]     for i in correct_amplitude_indices ])
        self.amplitude_weights            = np.array([ temp_amplitude_weights[i] for i in correct_amplitude_indices ])
        self.amplitude_first_eigenvectors = np.array([ temp_amplitude_vectors[i] for i in correct_amplitude_indices ])
        
        self.phase_eigenvalues        = np.array([ temp_phase_evs[i]     for i in correct_phase_indices ])
        self.phase_weights            = np.array([ temp_phase_weights[i] for i in correct_phase_indices ])
        self.phase_first_eigenvectors = np.array([ temp_phase_vectors[i] for i in correct_phase_indices ])
        
        for i in range(len(self.amplitude_first_eigenvectors)):
            if np.sum(self.amplitude_first_eigenvectors[i][:self.N]) < 0:
                self.amplitude_first_eigenvectors[i] *= -1
        for i in range(len(self.phase_first_eigenvectors)):
            if np.sum(self.phase_first_eigenvectors[i]) < 0:
                self.phase_first_eigenvectors[i] *= -1
        
        
                
        glimmering_amplitude_indices.sort()
        glimmering_phase_indices.sort()
        
        self.glimmering_amplitude_eigenvalues        = np.array([ temp_amplitude_evs[i]     for i in glimmering_amplitude_indices ])
        self.glimmering_amplitude_weights            = np.array([ temp_amplitude_weights[i] for i in glimmering_amplitude_indices ])
        self.glimmering_amplitude_first_eigenvectors = np.array([ temp_amplitude_vectors[i] for i in glimmering_amplitude_indices ])
        
        self.glimmering_phase_eigenvalues        = np.array([ temp_phase_evs[i]     for i in glimmering_phase_indices ])
        self.glimmering_phase_weights            = np.array([ temp_phase_weights[i] for i in glimmering_phase_indices ]) 
        self.glimmering_phase_first_eigenvectors = np.array([ temp_phase_vectors[i] for i in glimmering_phase_indices ])
        
        for i in range(len(self.glimmering_amplitude_first_eigenvectors)):
            if np.sum(self.glimmering_amplitude_first_eigenvectors[i][:self.N]) < 0:
                self.glimmering_amplitude_first_eigenvectors[i] *= -1
        for i in range(len(self.glimmering_phase_first_eigenvectors)):
            if np.sum(self.glimmering_phase_first_eigenvectors[i]) < 0:
                self.glimmering_phase_first_eigenvectors[i] *= -1
    
    def plot_phase(self, ax, which='all', label_energy=False, **plot_kwargs):
        if which == 'all':
            for i in range(len(self.phase_first_eigenvectors)):
                if label_energy:
                    plot_kwargs["label"] = f"{i+1} ({self.phase_eigenvalues[i]:.3f})"
                else:
                    plot_kwargs["label"] = f"{i+1}"
                __plot_line__(ax, self.epsilon_space, self.phase_first_eigenvectors[i], **plot_kwargs)
        else:
            if len(self.phase_first_eigenvectors) <= which:
                print(f"Warning: Tried to plot phase eigenvector {which}, but only {len(self.phase_first_eigenvectors)} available.")
            else:
                if label_energy:
                    plot_kwargs['label'] = f"{self.phase_eigenvalues[which]:.3f}"
                __plot_line__(ax, self.epsilon_space, self.phase_first_eigenvectors[which], **plot_kwargs)
    
    def plot_amplitude(self, axes, which='all', label_energy=False, **plot_kwargs):
        if which == 'all':
            for i in range(len(self.amplitude_first_eigenvectors)):
                if label_energy:
                    plot_kwargs["label"] = f"{i+1} ({self.amplitude_eigenvalues[i]:.3f})"
                else:
                    plot_kwargs["label"] = f"{i+1}"
                __plot_line__(axes[0], self.epsilon_space, self.amplitude_first_eigenvectors[i][:self.N], **plot_kwargs)
                __plot_line__(axes[1], self.epsilon_space, self.amplitude_first_eigenvectors[i][self.N:], **plot_kwargs)
        else:
            if len(self.amplitude_first_eigenvectors) <= which:
                print(f"Warning: Tried to plot amplitude eigenvector {which}, but only {len(self.amplitude_first_eigenvectors)} available.")
            else:
                if label_energy:
                    plot_kwargs['label'] = f"{self.amplitude_eigenvalues[which]:.3f}"
                __plot_line__(axes[0], self.epsilon_space, self.amplitude_first_eigenvectors[which][:self.N], **plot_kwargs)
                __plot_line__(axes[1], self.epsilon_space, self.amplitude_first_eigenvectors[which][self.N:], **plot_kwargs)
    
    def plot_glimmering_phase(self, ax, which='all', label_energy=False, **plot_kwargs):
        if which == 'all':
            for i in range(len(self.glimmering_phase_first_eigenvectors)):
                if label_energy:
                    plot_kwargs["label"] = f"{i+1} ({self.glimmering_phase_eigenvalues[i]:.3f})"
                else:
                    plot_kwargs["label"] = f"{i+1}"
                __plot_line__(ax, self.epsilon_space, self.glimmering_phase_first_eigenvectors[i], **plot_kwargs)
        else:
            if len(self.glimmering_phase_first_eigenvectors) <= which:
                print(f"Warning: Tried to plot glimmering phase eigenvector {which}, but only {len(self.glimmering_phase_first_eigenvectors)} available.")
            else:
                if label_energy:
                    plot_kwargs['label'] = f"{self.glimmering_phase_eigenvalues[which]:.3f}"
                __plot_line__(ax, self.epsilon_space, self.glimmering_phase_first_eigenvectors[which], **plot_kwargs)
    
    def plot_glimmering_amplitude(self, axes, which='all', label_energy=False, **plot_kwargs):
        if which == 'all':
            for i in range(len(self.glimmering_amplitude_first_eigenvectors)):
                if label_energy:
                    plot_kwargs["label"] = f"{i+1} ({self.glimmering_amplitude_eigenvalues[i]:.3f})"
                else:
                    plot_kwargs["label"] = f"{i+1}"
                __plot_line__(axes[0], self.epsilon_space, self.glimmering_amplitude_first_eigenvectors[i][:self.N], **plot_kwargs)
                __plot_line__(axes[1], self.epsilon_space, self.glimmering_amplitude_first_eigenvectors[i][self.N:], **plot_kwargs)
        else:
            if len(self.glimmering_amplitude_first_eigenvectors) <= which:
                print(f"Warning: Tried to plot glimmering amplitude eigenvector {which}, but only {len(self.glimmering_amplitude_first_eigenvectors)} available.")
            else:
                if label_energy:
                    plot_kwargs['label'] = f"{self.glimmering_amplitude_eigenvalues[which]:.3f}"
                __plot_line__(axes[0], self.epsilon_space, self.glimmering_amplitude_first_eigenvectors[which][:self.N], **plot_kwargs)
                __plot_line__(axes[1], self.epsilon_space, self.glimmering_amplitude_first_eigenvectors[which][self.N:], **plot_kwargs)