import numpy as np
import matplotlib.pyplot as plt
import lib.continued_fraction as cf
from lib.iterate_containers import *
from lib.extract_key import * 
# Calculates the resolvent in w^2

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

reversed = True

T = 0.0
U = 3.7
V = 1.0
name = f"T={T}/U={U}/V={V}"

folder = "data/modes/square/dos_900/"
fig, ax = plt.subplots()

name_suffix = "AFM"

data, data_real, w_lin, res = cf.resolvent_data(f"{folder}{name}", name_suffix, lower_edge=1., upper_edge=4, 
                                                number_of_values=20000, imaginary_offset=1e-6, xp_basis=True)
peak_pos_value = w_lin[np.argmax(data)]
print(peak_pos_value, data[np.argmax(data)])

import scipy.optimize as opt
def min_func(x):
    return res.continued_fraction(x + 1e-6j, True).imag

offset_peak = 0.2
search_bounds = (0 if peak_pos_value - offset_peak < 0 else peak_pos_value - offset_peak, 
                 np.sqrt(res.roots[0]) if peak_pos_value + offset_peak > np.sqrt(res.roots[0]) else peak_pos_value + offset_peak)

scipy_result = opt.fmin_l_bfgs_b(min_func, search_bounds[1] - 1e-2, bounds=[search_bounds], approx_grad=True, epsilon=1e-12)
peak_pos_value = scipy_result[0][0]
#print("Crude:", peak_pos_value, "\n")
print("SciPy:\n", scipy_result)
#print(min_func(peak_pos_value), min_func(scipy_test[0][0]), "\n")

data, data_real, w_lin, res = cf.resolvent_data_log_z(f"{folder}{name}", name_suffix, lower_edge=peak_pos_value,
                                                          range=0.01, begin_offset=1e-10,
                                                          number_of_values=2000, imaginary_offset=1e-6, xp_basis=True, reversed=reversed)

plot_data = np.log(np.abs(data_real))

ax.plot(w_lin, plot_data, label="Data", linewidth=1.75*plt.rcParams["lines.linewidth"])

from scipy.optimize import curve_fit
def func(x, a, b):
    return a * x + b

popt, pcov = curve_fit(func, w_lin, plot_data)
print(popt[0], " +/- ", np.sqrt(pcov[0][0]))
print(popt[1], " +/- ", np.sqrt(pcov[1][1]))
ax.text(0.05, 0.35, f"$a={popt[0]:.5f}$", transform = ax.transAxes)
ax.text(0.05, 0.3, f"$b={popt[1]:.5f}$", transform = ax.transAxes)
ax.plot(w_lin, func(w_lin, *popt), "k--", label="Fit")

plt.legend()
ax.set_xlabel(r"$\ln((z - z_0) / t)$")
ax.set_ylabel(r"$\ln(\Re G^\mathrm{ret}(z - z_0))$")
fig.tight_layout()

import os
plt.savefig(f"python/build/{os.path.basename(__file__).split('.')[0]}.pdf")
plt.show()
