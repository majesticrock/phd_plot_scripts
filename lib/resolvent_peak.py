import numpy as np
import scipy.optimize as opt
from lib.bounded_minimize import bounded_minimize
import lib.continued_fraction as cf

def linear_function(x, a, b):
    return a * x + b

class Peak:
    def __init__(self, data_folder, name_suffix, initial_search_bounds=(0., None), imaginary_offset=1e-6, xp_basis=True):
        self.data_folder = data_folder
        self.name_suffic = name_suffix
        self.imaginary_offset = imaginary_offset
        self.xp_basis = xp_basis
        
        data, data_real, w_lin, self.resolvent = cf.resolvent_data(data_folder, name_suffix, lower_edge=initial_search_bounds[0], upper_edge=initial_search_bounds[1], 
                                                             xp_basis=xp_basis, imaginary_offset=imaginary_offset, messages=False)
        self.peak_position = w_lin[np.argmax(data)]
    
    def imaginary_part(self, x):
        return self.resolvent.continued_fraction(x + self.imaginary_offset * 1j).imag
    
    def real_part(self, x):
        return self.resolvent.continued_fraction(x + self.imaginary_offset * 1j).real
    
    def improved_peak_position(self, xtol=2e-12):
        offset_peak = 0.2
        search_bounds = (0 if self.peak_position - offset_peak < 0 else self.peak_position - offset_peak, 
                 np.sqrt(self.resolvent.roots[0]) if self.peak_position + offset_peak > np.sqrt(self.resolvent.roots[0]) else self.peak_position + offset_peak)

        def min_func(x):
            return self.imaginary_part(x)
        
        result = bounded_minimize(min_func, bounds=search_bounds, xtol=xtol)
        self.peak_position = result["x"]
        return result
    
    def fit_real_part(self, range=0.01, begin_offset=1e-10, reversed=False):
        data, w_log = self.resolvent.data_log_z(lower_edge=self.peak_position, range=range, begin_offset=begin_offset, 
                                                    number_of_values=2000, imaginary_offset=self.imaginary_offset, reversed=reversed)
        # The absolute value is taken in case the real part is negative
        # This can be the case, depending on which side of the peak we are on
        # usually, if z>0 and if we are on the right side of the peak, real(data) > 0 and if we are left of the peaj real(data) < 0
        y_data = np.log(np.abs(data.real))
        self.popt, self.pcov = opt.curve_fit(linear_function, w_log, y_data)
        return self.popt, self.pcov, w_log, y_data
    
    def compute_weight(self):
        self.fit_real_part(begin_offset=1e-9, range=0.02)
        print(self.popt)
        return np.exp(self.popt[1]) / 2