import numpy as np
import matplotlib.pyplot as plt
import mrock_centralized_scripts.legacy_continued_fraction as cf
from mrock_centralized_scripts.iterate_containers import *
import mrock_centralized_scripts.plot_settings as ps

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

realPart = True
both = False

Ts = np.array([0.0])
Us = np.array([-2.5])
Vs = np.array([4., 6., 8., 10., 15., 25., 50.])

folder = "data/pre_pandas/modes/cube/dos_6000/"
fig, ax = plt.subplots()

if realPart or both:
    ax.set_xscale("log")
    ax.set_yscale("symlog")

plot_lower_lim = 20
plot_upper_lim = 620
name_suffix = "phase_SC"

realPlotter = ps.CURVEFAMILY(total_size(Ts, Us, Vs), axis=ax, allow_cycle=True)
realPlotter.set_individual_colors("default")

plotter = ps.CURVEFAMILY(total_size(Ts, Us, Vs), axis=ax, allow_cycle=True)
plotter.set_individual_colors("default")
plotter.set_individual_linestyles()

for T, U, V in iterate_containers(Ts, Us, Vs):
    name = f"T={T}/U={U}/V={V}"
    if len(Ts) > 1:
        label = f"$T={T}$"
    elif len(Us) > 1:
        label = f"$U={U}$"
    elif len(Vs) > 1:
        label = f"$V={V}$"
    
    data, data_real, w_lin, res = cf.resolvent_data(f"{folder}{name}", name_suffix, lower_edge=plot_lower_lim, upper_edge=plot_upper_lim, number_of_values=20000, imaginary_offset=1e-1, xp_basis=True)
    print(w_lin[np.argmax(data)])
    if realPart or both:
        realPlotter.plot(w_lin, data_real, label=label)
    if not realPart or both:
        plotter.plot(w_lin, data, label=label)

legend = plt.legend(loc=8)

ax.set_xlabel(r"$z / t$")
ax.set_ylabel(r"Spectral density / a.u.")

plt.show()
