import numpy as np
import matplotlib.pyplot as plt
import __path_appender as __ap
__ap.append()
import continued_fraction as cf

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

T = 0.
U = -2.5
V = -0.1

use_XP = True

folder = "data/pre_pandas/modes/cube/dos_6000/"
fig, ax = plt.subplots()

#ax.set_xscale("log")
#ax.set_yscale("log")

name = f"T={T}/U={U}/V={V}"
phase_imag, phase_real, w_log, res = cf.resolvent_data_log_z(f"{folder}{name}", "phase_SC", imaginary_offset=0, range=0.05, begin_offset=1e-3, number_of_values=10000, xp_basis=use_XP, ingore_first=5)
higgs_imag, higgs_real, w_log, res = cf.resolvent_data_log_z(f"{folder}{name}", "higgs_SC", imaginary_offset=0, range=0.05, begin_offset=1e-3, number_of_values=10000, xp_basis=use_XP, ingore_first=5)

diff_data = np.log(higgs_imag - phase_imag)
fit_data = diff_data
ax.plot(w_log, diff_data, linestyle="-", label="Higgs - Phase")
ax.plot(w_log, np.log(higgs_imag), linewidth=1.75*plt.rcParams["lines.linewidth"], linestyle=":", label="Higgs")

from scipy.optimize import curve_fit
def func(x, a, b):
    return a * x + b

popt, pcov = curve_fit(func, w_log, fit_data)
ax.text(0.05, 0.35, f"$a={popt[0]:.5f}$", transform = ax.transAxes)
ax.text(0.05, 0.3, f"$b={popt[1]:.5f}$", transform = ax.transAxes)
ax.plot(w_log, func(w_log, *popt), "k--", label="Fit")

legend = plt.legend()
ax.add_artist(legend)

ax.set_xlabel(r"$\ln((z - z_0) / t)$")
ax.set_ylabel(r"$\ln(-\Im G^\mathrm{ret}(z - z_0))$")
fig.tight_layout()

import os
#plt.savefig(f"python/build/{os.path.basename(__file__).split('.')[0]}.pdf")
plt.show()
