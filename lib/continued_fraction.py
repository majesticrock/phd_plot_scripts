import numpy as np
import gzip
import matplotlib.pyplot as plt

class ContinuedFraction:
    # In Python there is no need to declare one's variables beforehand.
    # How foolish of me to assume otherwise
    
    def terminator(self, w_param):
        w = w_param**2
        p = w - self.a_infinity
        q = 4 * self.b_infinity**2
        root = np.sqrt(np.real(p**2 - q), dtype=complex)
        
        if hasattr(w, '__len__'):
            return_arr = np.zeros(len(w), dtype=complex)
            for i in range(0, len(w)):
                if w_param[i].real > 0:
                    if w[i].real > self.roots[0]:
                        return_arr[i] = (p[i] - root[i]) / (2. * self.b_infinity**2)
                    else:
                        return_arr[i] = (p[i] + root[i]) / (2. * self.b_infinity**2)
                else:
                    if w[i].real > self.roots[1]:
                        return_arr[i] = (p[i] - root[i]) / (2. * self.b_infinity**2)
                    else:
                        return_arr[i] = (p[i] + root[i]) / (2. * self.b_infinity**2)
        else:
            if w_param.real > 0:
                if w.real > self.roots[0]:
                    return (p - root) / (2. * self.b_infinity**2)
                else:
                    return (p + root) / (2. * self.b_infinity**2)
            else:
                if w.real > self.roots[1]:
                    return (p - root) / (2. * self.b_infinity**2)
                else:
                    return (p + root) / (2. * self.b_infinity**2)
        return return_arr

    def continued_fraction(self, w_param, withTerminator = True):
        w = w_param**2
        if withTerminator:
            G = w - self.A[len(self.A) - self.terminate_at] - self.B[len(self.B) - self.terminate_at] * self.terminator(w_param.real)
        else:
            G = w - self.A[len(self.A) - self.terminate_at]
        for j in range(len(self.A) - self.terminate_at - 1, -1, -1):
            G = w - self.A[j] - self.B[j + 1] / G

        return self.B[0] / G
    
    def mark_continuum(self, axes=None):
        if axes is None:
            if self.z_squared:
                plt.axvspan(np.sqrt(self.roots[0]), np.sqrt(self.roots[1]), alpha=.2, color="purple", label="Continuum")
            else:
                plt.axvspan(self.roots[0], self.roots[1], alpha=.2, color="purple", label="Continuum")
        else:
            if self.z_squared:
                axes.axvspan(np.sqrt(self.roots[0]), np.sqrt(self.roots[1]), alpha=.2, color="purple", label="Continuum")
            else:
                axes.axvspan(self.roots[0], self.roots[1], alpha=.2, color="purple", label="Continuum")
    
    def __init__(self, data_folder, filename, z_squared=True, messages=True):
        self.z_squared = z_squared
        self.messages = messages
        file = f"{data_folder}/one_particle.dat.gz"
        with gzip.open(file, 'rt') as f_open:
            one_particle = np.abs(np.loadtxt(f_open).flatten())
        if z_squared:
            self.roots = np.array([np.min(one_particle) * 2, np.max(one_particle) * 2])**2
        else:
            self.roots = np.array([np.min(one_particle) * 2, np.max(one_particle) * 2])
        self.a_infinity = (self.roots[0] + self.roots[1]) * 0.5
        self.b_infinity = (self.roots[1] - self.roots[0]) * 0.25
        
        file = f"{data_folder}/{filename}.dat.gz"
        with gzip.open(file, 'rt') as f_open:
            M = np.loadtxt(f_open)
            self.A = M[0]
            self.B = M[1]
            
        deviation_from_infinity = np.zeros(len(self.A) - 1)
        for i in range(0, len(self.A) - 1):
            deviation_from_infinity[i] = abs((self.A[i] - self.a_infinity) / self.a_infinity) + abs((np.sqrt(self.B[i + 1]) - self.b_infinity) / self.b_infinity)
        # The lanczos coefficients have an oscillating behaviour at the beginnig
        # Thus there may be the best fit there by random chance, eventhough it isn't really converged yet
        # Therefore, we omit the first n (10) data points from our best fit search
        ingore_first = 10
        best_approx = np.argmin(deviation_from_infinity[ingore_first:]) + ingore_first
        self.terminate_at = len(self.A) - best_approx
        if self.messages: 
            print("Terminating at i =", best_approx)

def continuum_edges(data_folder, name_suffix, xp_basis=False):
    if xp_basis:
        if name_suffix == "AFM" or name_suffix == "CDW":
            res = ContinuedFraction(data_folder, f"resolvent_higgs_{name_suffix}", True, False)
        else:
            res = ContinuedFraction(data_folder, f"resolvent_{name_suffix}", True, False)
    else:
        res = ContinuedFraction(data_folder, f"resolvent_{name_suffix}_a", True, False)
    
    return np.sqrt(res.roots)

def resolvent_data_log_z(data_folder, name_suffix, lower_edge=None, range=None, begin_offset=1e-6, xp_basis=False, number_of_values=20000, imaginary_offset=1e-6, withTerminator=True, reversed=False, messages=True):
    edges = continuum_edges(data_folder, name_suffix, xp_basis)
        
    if lower_edge is None:
        lower_edge = edges[0]
    if range is None:
        upper_range = np.log(edges[1])
    else:
        upper_range = np.log(begin_offset + range)
    lower_range = np.log(begin_offset)
    
    w_log = np.linspace(lower_range, upper_range, number_of_values, dtype=complex)
    w_log += (imaginary_offset * 1j)
    if reversed:
        w_usage = lower_edge - np.exp(w_log)
    else:
        w_usage = lower_edge + np.exp(w_log)
    
    data = np.zeros(number_of_values, dtype=complex)
    if xp_basis:
        if name_suffix == "AFM" or name_suffix == "CDW":
            res = ContinuedFraction(data_folder, f"resolvent_higgs_{name_suffix}", True, messages)
        else:
            res = ContinuedFraction(data_folder, f"resolvent_{name_suffix}", True, messages)
        
        data = res.continued_fraction( w_usage, withTerminator)
    else:
        element_names = ["a", "a+b", "a+ib"]
        res = ContinuedFraction(data_folder, f"resolvent_{name_suffix}_{element_names[0]}", True, messages)
        
        for idx, element in enumerate(element_names):
            res = ContinuedFraction(data_folder, f"resolvent_{name_suffix}_{element}", True, messages)
            
            def dos(w):
                if idx==0:
                    return res.continued_fraction(w, withTerminator)
                else:
                    return np.sqrt(w) * res.continued_fraction(w, withTerminator)
                    
            if idx == 0:
                data += dos( w_usage )
            elif idx == 1:
                data -= dos( w_usage )
            elif idx == 2:
                data += dos( w_usage )
            
    return -data.imag, data.real, w_log.real, res

def resolvent_data(data_folder, name_suffix, lower_edge, upper_edge=None, xp_basis=False, number_of_values=20000, imaginary_offset=1e-6, withTerminator=True, use_start=True, messages=True):
    data = np.zeros(number_of_values, dtype=complex)
    if xp_basis:
        if name_suffix == "AFM" or name_suffix == "CDW":
            res = ContinuedFraction(data_folder, f"resolvent_higgs_{name_suffix}", True, messages)
        else:
            res = ContinuedFraction(data_folder, f"resolvent_{name_suffix}", True, messages)
        
        if upper_edge is None:
            upper_edge = np.sqrt(res.roots[1]) + 0.5
        w_lin = np.linspace(lower_edge, upper_edge, number_of_values, dtype=complex)[0 if use_start else 1:]
        w_lin += (imaginary_offset * 1j)
        data = res.continued_fraction( w_lin, withTerminator)
    else:
        element_names = ["a", "a+b", "a+ib"]
        res = ContinuedFraction(data_folder, f"resolvent_{name_suffix}_{element_names[0]}", True, messages)
        if upper_edge is None:
            upper_edge = np.sqrt(res.roots[1]) + 0.5
        w_lin = np.linspace(lower_edge, upper_edge, number_of_values, dtype=complex)[0 if use_start else 1:]
        w_lin += (imaginary_offset * 1j)
        
        for idx, element in enumerate(element_names):
            res = ContinuedFraction(data_folder, f"resolvent_{name_suffix}_{element}", True, messages)
            
            def dos(w):
                if idx==0:
                    return res.continued_fraction(w, withTerminator)
                else:
                    return np.sqrt(w) * res.continued_fraction(w, withTerminator)
                    
            if idx == 0:
                data += dos( w_lin )
            elif idx == 1:
                data -= dos( w_lin )
            elif idx == 2:
                data += dos( w_lin )
            
    return -data.imag, data.real, w_lin.real, res

def resolvent_in_continuum(data_folder, name_suffix, range=None, xp_basis=False, number_of_values=20000, imaginary_offset=1e-6, withTerminator=True, w_space=np.linspace):
    borders = continuum_edges(data_folder, name_suffix, xp_basis)
    if range is None:
        return resolvent_data(data_folder, name_suffix, borders[0], borders[1], xp_basis, number_of_values, imaginary_offset, withTerminator, w_space)
    else:
        return resolvent_data(data_folder, name_suffix, borders[0], borders[0] + range, xp_basis, number_of_values, imaginary_offset, withTerminator, w_space)