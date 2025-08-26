import numpy as np

class Mode:
    def __init__(self, first_x, first_energy, first_weight, first_weight_error):
        self.x = [first_x]
        self.energies = [first_energy]
        self.weights = [first_weight]
        self.weight_errors = [first_weight_error]
        
    def append(self, new_x, new_energy, weight, weight_error):
        self.x.append(new_x)
        self.energies.append(new_energy)
        self.weights.append(weight)
        self.weight_errors.append(weight_error)
        
class ModeCollector:
    def __init__(self, first_x, first_energies, first_weights, first_weight_errors):
        self.modes = [ Mode(first_x, energy, weight, weight_error) for energy, weight, weight_error in zip(first_energies, first_weights, first_weight_errors) if energy is not None]
    
    def append_new_energy(self, new_x, new_energies, new_weights, new_weight_errors):
        EPS = 0.01
        for new_energy, weight, weight_error in zip(new_energies, new_weights, new_weight_errors):
            if new_energy is None:
                continue
            best_mode_idx = None
            best_energy_diff = 10000
            for j, mode in enumerate(self.modes):
                if np.abs(mode.energies[-1] - new_energy) < best_energy_diff:
                    best_energy_diff = np.abs(mode.energies[-1] - new_energy)
                    best_mode_idx = j
            if best_mode_idx is None or best_energy_diff > EPS:
                self.modes.append(Mode(new_x, new_energy, weight, weight_error))
            else:
                self.modes[best_mode_idx].append(new_x, new_energy, weight, weight_error)